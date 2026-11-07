"""Card 116.1: SlotContrast-style slot model on frozen DINOv2 patches at 448 px (32 x 32 = 1,024 patches).

SlotContrast (2412.14295, MOVi-C settings): MLP adapter on the frozen features; slot attention with K = 8 slots of 64
numbers from a learned fixed initialisation (3 iterations on the first frame, 2 after); a 1-layer transformer (4 heads)
carries slots to the next model step; an MLP decoder (DINOSAUR) gives each slot a feature and an alpha per patch, the
alphas softmaxed over slots are the masks; loss = squared error of the reconstructed frozen features + 0.5 x the
batch slot-slot contrastive loss (temperature 0.1). Segments of 4 model steps (5 environment steps apart), Adam 4e-4,
2,500 warm-up steps, exponential decay, gradient clipping 0.05. No actions.

  bin/prun python tools/card116/slots.py --minutes 30 [--hidden 512] [--batch 32] [--out slots.pt] → runs/116/<out>

Card 116.3 adds: --feat feat256 (the 256-px renders' features), --K 32 slots, and --teacher parts_train.npz: half of
each batch are SAM 2-labelled segments, whose masks get WorldDP's Tversky loss (alpha 0.99 on missed patches, beta
0.01 on extra ones, each segment weighted by (1/size)^0.2, normalised over the batch) after Hungarian matching of
segments to slots in each frame. 10% of labelled segments are held out to watch the mask loss.
"""
from scipy.optimize import linear_sum_assignment
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/116'
EP, T = 1001, 4
arg = lambda k, d: next((sys.argv[i + 1] for i in range(len(sys.argv) - 1) if sys.argv[i] == k), d)


class SlotAttention(nn.Module):
    def __init__(self, d_in=128, d=64, eps=1e-8):
        super().__init__()
        self.ni, self.ns, self.nm = nn.LayerNorm(d_in), nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d_in, d, bias=False), nn.Linear(d_in, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.d, self.eps = d, eps

    def forward(self, slots, x, iters):
        x = self.ni(x)
        k, v = self.k(x), self.v(x)                                   # [B, N, d]
        for _ in range(iters):
            q = self.q(self.ns(slots))                                # [B, K, d]
            logits = torch.einsum('bkd,bnd->bkn', q, k) / math.sqrt(self.d)
            attn = logits.softmax(1) + self.eps                       # over slots
            w = attn / attn.sum(-1, keepdim=True)                     # weighted mean over inputs
            upd = torch.einsum('bkn,bnd->bkd', w, v)
            B, K, d = slots.shape
            slots = self.gru(upd.reshape(B * K, d), slots.reshape(B * K, d)).reshape(B, K, d)
            slots = slots + self.mlp(self.nm(slots))
        return slots, attn


class SlotModel(nn.Module):
    def __init__(self, K=8, d=64, n=1024, feat=384, hidden=512):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, 2 * feat), nn.GELU(), nn.Linear(2 * feat, 128))
        self.sa = SlotAttention(128, d)
        self.init = nn.Parameter(torch.randn(1, K, d) * 0.1)
        self.pred = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.pos = nn.Parameter(torch.randn(1, 1, n, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, feat + 1))

    def decode(self, slots):                                          # [B, K, d] → recon [B, N, F], masks [B, K, N]
        out = self.dec(slots[:, :, None] + self.pos)                  # [B, K, N, F+1]
        masks = out[..., -1].softmax(1)
        return torch.einsum('bkn,bknf->bnf', masks, out[..., :-1]), masks

    def forward(self, feats):                                         # feats [B, T, N, F] → slots [B,T,K,d], recon, masks
        B, Tn = feats.shape[:2]
        x = self.adapter(feats)
        slots, out_s, rec, msk = self.init.expand(B, -1, -1), [], [], []
        for t in range(Tn):
            start = slots if t == 0 else self.pred(slots)
            slots, _ = self.sa(start, x[:, t], 3 if t == 0 else 2)
            r, m = self.decode(slots)
            out_s.append(slots); rec.append(r); msk.append(m)
        return torch.stack(out_s, 1), torch.stack(rec, 1), torch.stack(msk, 1)


def contrast(s0, s1, tau=0.1):
    """Batch slot-slot contrast: slot k of video b at t-1 against every slot of every video at t."""
    a = F.normalize(s0.reshape(-1, s0.shape[-1]), dim=-1)
    b = F.normalize(s1.reshape(-1, s1.shape[-1]), dim=-1)
    return F.cross_entropy(a @ b.T / tau, torch.arange(len(a), device=a.device))


def tversky(masks, labels, alpha=0.99, beta=0.01, eps=1e-6):
    """masks [F, K, N] (soft), labels [F, N] (segment ids, -1 none) → weighted Tversky loss of the matched pairs."""
    Fn, Kn, N = masks.shape
    M = int(labels.max().item()) + 1
    G = F.one_hot(labels.clamp(min=0).long(), M).float() * (labels >= 0)[..., None].float()   # [F, N, M]
    size = G.sum(1)                                                                            # [F, M]
    tp = torch.einsum('fkn,fnm->fkm', masks, G)
    fn = size[:, None] - tp
    fp = masks.sum(-1)[..., None] - tp
    cost = 1 - (tp + eps) / (tp + alpha * fn + beta * fp + eps)                              # [F, K, M]
    c = cost.detach().float().cpu().numpy()
    sz = size.cpu().numpy()
    fi, ki, mi = [], [], []
    for f in range(Fn):
        valid = np.where(sz[f] > 0)[0]
        r, col = linear_sum_assignment(c[f][:, valid])
        fi += [f] * len(r); ki += list(r); mi += list(valid[col])
    fi, ki, mi = (torch.tensor(a, device=masks.device) for a in (fi, ki, mi))
    w = size[fi, mi] ** -0.2
    return (cost[fi, ki, mi] * w).sum() / w.sum()


def load_model(path=OUT / 'slots.pt', device='cuda'):
    ck = torch.load(path, map_location=device)
    m = SlotModel(K=ck.get('K', 8), hidden=ck['hidden']).to(device)
    m.load_state_dict(ck['state'])
    return m.eval()


def main():
    minutes = float(arg('--minutes', 30))
    hidden, B = int(arg('--hidden', 512)), int(arg('--batch', 32))
    K, lam = int(arg('--K', 8)), float(arg('--lam', 1.0))
    dev = 'cuda'
    torch.manual_seed(0)
    feat = np.load(OUT / f"{arg('--feat', 'feat448')}_train.npy", mmap_mode='r')
    frames = np.load(OUT / 'frames448_train.npy')
    model = SlotModel(K=K, hidden=hidden).to(dev)
    teach = arg('--teacher', None)
    if teach:
        z = np.load(OUT / teach)
        n_hold = max(1, len(z['rows']) // 10)
        rows_all = z['rows'][:, None] + np.arange(T)[None]                                # [n, T] feature rows
        order = np.argsort(rows_all.ravel())
        lf = np.empty((rows_all.size, 1024, 384), np.float16)
        lf[order] = feat[rows_all.ravel()[order]]                                  # sorted reads from the memmap
        lfeat = torch.from_numpy(lf.reshape(len(rows_all), T, 1024, 384)).to(dev)
        llab = torch.from_numpy(z['labels'].astype(np.int64)).to(dev)
        hold, train_l = torch.arange(n_hold, device=dev), torch.arange(n_hold, len(rows_all), device=dev)
        print(f'teacher: {len(train_l)} labelled segments for training, {n_hold} held out', flush=True)
        B = B // 2
    print('parameters', round(sum(p.numel() for p in model.parameters()) / 1e6, 2), 'M', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=4e-4)
    CH = 2048
    starts = np.arange(0, len(frames), CH)
    rng = np.random.default_rng(0)
    t0, step, budget, done, log = time.time(), 0, minutes * 60, False, []
    while not done:
        rng.shuffle(starts)
        for s0 in starts:
            sl = slice(s0, min(s0 + CH + T, len(frames)))
            f = torch.from_numpy(np.asarray(feat[sl])).to(dev)
            fr = frames[sl]
            i = np.arange(len(fr) - T + 1)
            i = i[fr[i] // EP == fr[i + T - 1] // EP]
            for b in torch.from_numpy(rng.permutation(i)).to(dev).split(B):
                frac = (time.time() - t0) / budget
                if frac >= 1:
                    done = True
                    break
                lr = 4e-4 * min(1, (step + 1) / 2500) * 0.5 ** frac
                for g in opt.param_groups:
                    g['lr'] = lr
                x = f[b[:, None] + torch.arange(T, device=dev)].float()
                if teach:
                    li = train_l[torch.randint(len(train_l), (B,), device=dev)]
                    x = torch.cat([x, lfeat[li].float()])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    slots, rec, msk = model(x)
                    rl = F.mse_loss(rec.float(), x)
                    cl = sum(contrast(slots[:, t - 1].float(), slots[:, t].float()) for t in range(1, T)) / (T - 1)
                    loss = rl + 0.5 * cl
                ml = torch.zeros(())
                if teach:
                    nb = len(li)
                    ml = tversky(msk[-nb:].float().reshape(nb * T, K, -1), llab[li].reshape(nb * T, -1))
                    loss = loss + lam * ml
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05)
                opt.step()
                step += 1
                if step % 250 == 0:
                    hl = 0.0
                    if teach and step % 1000 == 0:
                        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                            hs = hold[:64]
                            _, _, hm = model(lfeat[hs].float())
                        hl = tversky(hm.float().reshape(len(hs) * T, K, -1), llab[hs].reshape(len(hs) * T, -1)).item()
                    log.append((step, time.time() - t0, rl.item(), cl.item(), ml.item(), hl))
                    print(f'step {step} {time.time() - t0:.0f}s lr {lr:.2e} rec {rl.item():.4f} contrast {cl.item():.3f}'
                          f' mask {ml.item():.3f}' + (f' held-out mask {hl:.3f}' if hl else ''), flush=True)
            if done:
                break
    torch.save({'state': model.state_dict(), 'hidden': hidden, 'K': K, 'steps': step, 'log': log}, OUT / arg('--out', 'slots.pt'))
    print(f'final: {step} steps in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
