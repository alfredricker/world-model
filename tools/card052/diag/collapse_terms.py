import sys; sys.path.insert(0,'tools/card052')
import numpy as np, torch
import effect as EF
DR=EF.DR; GN=DR.GN
GN.TRAIN[:]=["red","green","blue"]; GN.TINT=0; GN.NOISE=0
fn=torch.nn.functional
s=EF.Stream(399); tiles=[]
for _ in range(3000):
    t,_=s.step(); tiles+=t
for mode, lr, terms in [("raw",1e-3,"all"),("norm",1e-4,"all"),("norm",1e-3,"scale+centre"),("norm",1e-3,"shape")]:
    DR.Encoder.ANCHOR="visreg"; DR.Encoder.CODEBOOK="fixed"
    enc=EF.Encoder(399)
    rng=np.random.default_rng(0)
    opt=torch.optim.Adam(enc.enc.parameters(), lr=lr)
    acts={}
    h=enc.enc[3].register_forward_hook(lambda m,i,o: acts.__setitem__('r', (o>0).float().mean().item()))
    for it in range(601):
        b=[tiles[i] for i in rng.integers(len(tiles), size=256)]
        x=enc.x(b); raw=enc.enc(x); z=fn.normalize(raw.reshape(len(x),4,8),dim=-1)
        v = raw if mode=="raw" else z.reshape(len(x),-1)
        tgt = 1.0 if mode=="raw" else 8**-0.5
        mu=v.mean(0); zc=v-mu; std=zc.std(0,unbiased=False)
        p=(zc/(std.detach()+1e-6))@fn.normalize(torch.randn(v.shape[1],64,device=v.device),dim=0)
        n=len(v); q=torch.distributions.Normal(0.,1.).icdf(torch.arange(1,n+1,device=v.device)/(n+1))
        Lc=mu.pow(2).mean(); Ls=(tgt-std).pow(2).mean(); Lsh=(torch.sort(p,0).values-q[:,None]).pow(2).mean()
        loss={"all":Lc+Ls+Lsh,"scale+centre":Lc+Ls,"shape":Lsh}[terms]
        opt.zero_grad(); loss.backward(); opt.step()
        if it%200==0:
            _,idx=enc.quant(z)
            print(f"{mode} lr {lr} {terms}: it {it} centre {Lc.item():.3f} scale {Ls.item():.3f} shape {Lsh.item():.3f} std {std.mean().item():.4f} relu-on {acts['r']:.3f} codes/part {[len(set(idx[:,k].tolist())) for k in range(4)]}")
    h.remove()
