"""Card 111.1, after the run: lamps lighting, closed answers by how many cases they rest on, and whether complete.
Run from the repo root: bin/prun python tools/card111.1/closed_check.py"""
import sys, numpy as np
from collections import Counter
sys.argv=["x","labels"]; sys.path.insert(0,"tools/card111.1")
import world1 as M, contrast as CT
z=np.load("runs/card102/world1_test.npz"); ztr=np.load("runs/card102/world1_train.npz")
rng=np.random.default_rng(1021); M.W1.PER=300; ev=M.W1.events(z,rng)
sem=z["sem"].reshape(-1,63).astype(np.int64); semtr=ztr["sem"].reshape(-1,63).astype(np.int64)
act=z["act"].astype(np.int64); acttr=ztr["act"].astype(np.int64)
T=np.flatnonzero(np.r_[ztr["ep"][1:]==ztr["ep"][:-1],False]&~ztr["done"])
W,H=9,7; names="fl wa ag bo la li ou".split()
def producible(g):
    for y,x in zip(*np.nonzero((g==4)|(g==5))):
        near=any(0<=y+dy<H and 0<=x+dx<W and g[y+dy,x+dx]==3 for dx,dy in M.W1.MOVES.values())
        if near!=(g[y,x]==5): return False
    return True
cache={}; left=Counter(); ok_left=Counter(); shown=0; facts=Counter()
for t,i in ev["lamp lights"]:
    a=int(act[t]); c0=int(sem[t,i]); c1=int(sem[t+1,i])
    if (a,c0,c1) not in cache:
        ts=T[acttr[T]==a]; tt,ii=np.nonzero(semtr[ts]==c0); tt=ts[tt]
        cache.clear(); cache[(a,c0,c1)]=CT.Situation(M.windows(semtr,tt,ii,6),semtr[tt+1,ii]==c1,6)
    sit=cache[(a,c0,c1)]
    e=M.windows(sem,np.array([t]),np.array([i]),6)[0]
    g=sem[t].reshape(H,W); base=M.W1.true_next(g,a).reshape(-1)[i]
    present=[k for k in (0,1,3,4,5) if (sem[t]==k).any()]
    tc=set()
    for j in range(63):
        if j==i: continue
        for k in ([0] if sem[t,j]==2 else [k for k in present if k!=sem[t,j]]):
            g2=g.copy().reshape(-1); g2[j]=k; g2=g2.reshape(H,W)
            if producible(g2) and M.W1.true_next(g2,a).reshape(-1)[i]!=base: tc.add(j)
    fin=[]; C=CT.greedy(sit,e,None,fin); ans=fin[2]
    if not ans["closed"]: continue
    got=set()
    for f in C:
        for c in f[7]:
            y,x=i//9+c//17-6,i%9+c%17-8
            if 0<=y<H and 0<=x<W: got.add(y*9+x)
    complete=tc<=got
    b="<10" if ans["items left"]<10 else "10-29" if ans["items left"]<30 else "30+"
    left[b]+=1; ok_left[b]+=int(complete)
    if not complete:
        facts[" & ".join(f"({f[0]//17-6},{f[0]%17-8}){'=' if f[2] else '!='}{names[f[1]]}" for f in C)]+=1
        if shown<2:
            shown+=1; print("lamp",(i//9,i%9),"action",a,"items left",ans["items left"],"missed",[(j//9-i//9,j%9-i%9) for j in sorted(tc-got)])
            print("\n".join("   "+" ".join(names[v] for v in r) for r in g))
            print("  kept",[((f[0]//17-6,f[0]%17-8),("=" if f[2] else "!=")+names[f[1]],f[5]) for f in C])
print("closed answers by items left:",dict(left),"complete:",dict(ok_left))
print("most common kept facts in closed-but-incomplete:",facts.most_common(4))
