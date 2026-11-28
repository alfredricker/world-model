"""Card 174: 100 episodes never used before (900-999 of the play file; card 172 trained on 0-449), built as card 172's.
  build_fresh.py -> runs/174/meta_fresh.npz, runs/174/feat_fresh.npy  fp16 [100, 201, 1024, 64] (paced writes)"""
import os, time, resource
import numpy as np
import torch
import build as B                                     # card 172

torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", 0.5)))
OUT, A, N = B.ROOT / "runs/174", 900, 100
z = np.load(B.DATA / B.FILE["train"])
assert len(z["qpos"]) >= (A + N) * B.EP, len(z["qpos"])
rows = ((A + np.arange(N))[:, None] * B.EP + B.STEPS[None]).reshape(-1)
o, q = z["observations"][rows], z["qpos"][rows]
np.savez(OUT / "meta_fresh.npz", pose=np.c_[o[:, 0:6], o[:, 12:18]].reshape(N, len(B.STEPS), 12).astype(np.float32),
         cube=q[:, 14:17].reshape(N, len(B.STEPS), 3), drawer=np.abs(q[:, 23]).reshape(N, -1), window=np.abs(q[:, 24]).reshape(N, -1),
         locks=z["button_states"][rows].reshape(N, len(B.STEPS), 2))
from train import features
from encode import load_encoder
render, enc, t0 = B.renderer(), load_encoder("cuda"), time.time()
p = torch.load(B.ROOT / "runs/132/pca.pt", map_location="cuda", weights_only=False)
mu, sd, V, scale = (p[k].float() for k in ("mu", "sd", "V", "scale"))
qpos, btn = z["qpos"], z["button_states"]
X = np.lib.format.open_memmap(OUT / "feat_fresh.npy", "w+", np.float16, (N, len(B.STEPS), 1024, B.K))
for i in range(N):
    r = (A + i) * B.EP + B.STEPS
    fr = torch.from_numpy(render(qpos[r], btn[r])).cuda()
    with torch.no_grad():
        f = torch.cat([(((features(enc, fr[k:k + 67]) - mu) / sd) @ V) / scale for k in range(0, len(r), 67)])
    X[i] = f.half().cpu().numpy()
    if i % 10 == 9:
        X.flush(); time.sleep(1)
X.flush()
print(f"fresh done {time.time() - t0:.0f}s | peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GB, peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.2f} GB", flush=True)
os._exit(0)
