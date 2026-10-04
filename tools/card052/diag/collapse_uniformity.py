import sys; sys.path.insert(0,'tools/card052')
import numpy as np, torch
import effect as EF
DR=EF.DR; GN=DR.GN
GN.TRAIN[:]=["red","green","blue"]; GN.TINT=0; GN.NOISE=0
fn=torch.nn.functional
s=EF.Stream(399); tiles=[]
for _ in range(3000):
    t,_=s.step(); tiles+=t
U=np.unique(np.stack(tiles).reshape(len(tiles),-1),axis=0).reshape(-1,8,8,3)
def unif(z, where):
    if where=="part":   # per part, on its sphere
        return sum(torch.pdist(z[:,k]).pow(2).mul(-2).exp().mean().log() for k in range(z.shape[1]))/z.shape[1]
    v=fn.normalize(z.reshape(len(z),-1),dim=-1)
    return torch.pdist(v).pow(2).mul(-2).exp().mean().log()
for where in ("part","whole"):
  for book in ("learned","fixed"):
    DR.Encoder.ANCHOR="visreg"; DR.Encoder.CODEBOOK=book
    enc=EF.Encoder(399)
    rng=np.random.default_rng(0)
    for it in range(1501):
        b=[tiles[i] for i in rng.integers(len(tiles), size=256)]
        x=enc.x(b); z=enc.pieces(x); zq,idx=enc.quant(z)
        loss=unif(z,where)+0.25*fn.mse_loss(z,zq.detach())
        if book=="learned": loss=loss+fn.mse_loss(zq,z.detach())
        enc.opt.zero_grad(); loss.backward(); enc.opt.step()
        if it in (500,1500):
            with torch.no_grad():
                zu=enc.pieces(enc.x(U)); _,iu=enc.quant(zu)
                dmin=torch.pdist(zu.reshape(len(U),-1)).min().item()
            tu=len(set(map(tuple,iu.cpu().numpy().tolist())))
            print(f"uniformity {where:5s} {book:7s} it {it}: tuples for {len(U)} distinct tiles {tu}, codes/part {[len(set(iu[:,k].tolist())) for k in range(4)]}, closest pair {dmin:.3f}")
