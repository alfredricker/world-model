"""Card 111.1, after the run: for a lamp going out, each true condition the greedy search missed, tested with every
other true condition held at the event's value (and, with argument hold, also the box pushed away, which the truth
leaves unjudged). Run from the repo root: bin/prun python tools/card111.1/oracle_check.py no|hold"""
import sys, math, numpy as np, torch
from collections import Counter
sys.argv_hold = sys.argv[1]=="hold"; sys.argv=["x","labels"]; sys.path.insert(0,"tools/card111.1")
import world1 as M, contrast as CT
z=np.load("runs/card102/world1_test.npz"); ztr=np.load("runs/card102/world1_train.npz")
rng=np.random.default_rng(1021); M.W1.PER=300; ev=M.W1.events(z,rng)
sem=z["sem"].reshape(-1,63).astype(np.int64); semtr=ztr["sem"].reshape(-1,63).astype(np.int64)
act=z["act"].astype(np.int64); acttr=ztr["act"].astype(np.int64)
T=np.flatnonzero(np.r_[ztr["ep"][1:]==ztr["ep"][:-1],False]&~ztr["done"])
W,H=9,7
def producible(g):
    for y,x in zip(*np.nonzero((g==4)|(g==5))):
        near=any(0<=y+dy<H and 0<=x+dx<W and g[y+dy,x+dx]==3 for dx,dy in M.W1.MOVES.values())
        if near!=(g[y,x]==5): return False
    return True
d=lambda x: torch.tensor([x],device="cuda",dtype=torch.float64)
def stats(mask,col,y):
    N=int(mask.sum()); S=int((mask&y).sum()); n=int((mask&col).sum()); k=int((mask&col&y).sum())
    return N,S,n,k,round(float(CT.fisher_upper(d(k),d(N),d(S),d(n))[0])/math.log(10),1)
HOLD=sys.argv_hold; cache={}; shown=0; kinds=Counter(); contr=[0,0,0]; tot=[0]; unjs=Counter()
for t,i in ev["lamp goes out"]:
    a=int(act[t]); c0=int(sem[t,i]); c1=int(sem[t+1,i])
    if (a,c0,c1) not in cache:
        ts=T[acttr[T]==a]; tt,ii=np.nonzero(semtr[ts]==c0); tt=ts[tt]
        cache.clear(); cache[(a,c0,c1)]=(M.windows(semtr,tt,ii,6),CT.Situation(M.windows(semtr,tt,ii,6),semtr[tt+1,ii]==c1,6))
    vals,sit=cache[(a,c0,c1)]
    e=M.windows(sem,np.array([t]),np.array([i]),6)[0]
    g=sem[t].reshape(H,W); base=M.W1.true_next(g,a).reshape(-1)[i]
    present=[k for k in (0,1,3,4,5) if (sem[t]==k).any()]
    tc=[]; unj=[]
    for j in range(63):
        if j==i: continue
        hit=False; jud=False
        for k in ([0] if sem[t,j]==2 else [k for k in present if k!=sem[t,j]]):
            g2=g.copy().reshape(-1); g2[j]=k; g2=g2.reshape(H,W)
            if producible(g2):
                jud=True
                if M.W1.true_next(g2,a).reshape(-1)[i]!=base: hit=True
        if hit: tc.append(j)
        if not jud: unj.append(j)
    cells=[int((j//W-i//W+6)*17+(j%W-i%W+8)) for j in tc]
    hold_extra=[int((j//W-i//W+6)*17+(j%W-i%W+8)) for j in unj] if HOLD else []
    unjs.update(sem[t,j] for j in unj)
    fin=[]; C=CT.greedy(sit,e,None,fin)
    got=set(c for f in C for c in f[7])
    y=sit.y.bool(); m=fin[0].bool()
    for c in cells:
        tot[0]+=1
        if c in got:
            col=sit.M[:,c*sit.K+int(e[c])].bool(); held=torch.ones_like(y)
            for o in cells+hold_extra:
                if o!=c: held&=sit.M[:,o*sit.K+int(e[o])].bool()
            h=stats(held,col,y)
            if h[2]>=5 and h[0]-h[2]>=5: contr[0]+=1; contr[2]+=1
            continue
        col=sit.M[:,c*sit.K+int(e[c])].bool()
        held=torch.ones_like(y)
        for o in cells+hold_extra:
            if o!=c: held&=sit.M[:,o*sit.K+int(e[o])].bool()
        h=stats(held,col,y)
        if h[2]>=5 and h[0]-h[2]>=5:
            contr[0]+=1; contr[1]+=int(h[4]<math.log10(0.01/50))
        if h[2]>=5 and h[0]-h[2]>=5 and h[4]<math.log10(0.01/50):
            g_=stats(m,col,y)
            kinds[(c//17-6,c%17-8, int(e[c]))]+=1
            if shown<0:
                shown+=1
                print("event lamp",(i//9,i%9),"action",a,"missed cell",(c//17-6,c%17-8),"value",int(e[c]))
                print("  kept:",[((f[0]//17-6,f[0]%17-8),f[1],"=" if f[2] else "!=") for f in C], "answer", fin[2]["closed"], fin[2]["items left"], fin[2]["failures left"])
                print("  greedy final subset (N, successes, n with e's value, successes among them, log10 p):",g_)
                print("  other true conditions held (same):",h)
print("missed contrastable conditions, by (dy, dx, value):",kinds.most_common(8))
print("hold unjudged cells:",HOLD,"| true conditions",tot[0],"contrastable",contr[0],"found among contrastable",contr[2],"missed contrastable & oracle-significant",sum(kinds.values()),"| unjudged kinds",dict(unjs))
