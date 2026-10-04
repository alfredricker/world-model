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
print('distinct tiles', len(U))
def reg(v, kind, tgt):
    mu=v.mean(0); zc=v-mu; std=zc.std(0,unbiased=False)
    if kind=="scale":                     # VISReg without its shape term
        return mu.pow(2).mean()+(tgt-std).pow(2).mean()
    c=(zc.T@zc)/(len(v)-1); off=c-torch.diag(torch.diag(c))
    return torch.relu(tgt-std).mean()+off.pow(2).sum()/v.shape[1]   # VICReg variance (hinge) + covariance
for kind in ("scale","vicreg"):
  for book in ("learned","fixed"):
    DR.Encoder.ANCHOR="visreg"; DR.Encoder.CODEBOOK=book
    enc=EF.Encoder(399)
    rng=np.random.default_rng(0)
    for it in range(1501):
        b=[tiles[i] for i in rng.integers(len(tiles), size=256)]
        x=enc.x(b); z=enc.pieces(x); zq,idx=enc.quant(z)
        loss=reg(z.reshape(len(x),-1),kind,8**-0.5)+0.25*fn.mse_loss(z,zq.detach())
        if book=="learned": loss=loss+fn.mse_loss(zq,z.detach())
        enc.opt.zero_grad(); loss.backward(); enc.opt.step()
        if it in (500,1500):
            with torch.no_grad():
                zu=enc.pieces(enc.x(U)); _,iu=enc.quant(zu)
            tu=len(set(map(tuple,iu.cpu().numpy().tolist())))
            print(f"{kind:6s} {book:7s} it {it}: tuples for {len(U)} distinct tiles {tu}, codes/part {[len(set(iu[:,k].tolist())) for k in range(4)]}")
