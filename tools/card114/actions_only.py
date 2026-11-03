"""Card 114: the action-only baseline (added after the run). Gradient-boosted trees on the action chunks the
predictor sees, trained on pulls of episodes 0-899, tested on the held-out pulls.

  bin/prun python tools/card114/actions_only.py
"""
import sys, numpy as np
sys.path.insert(0,'tools/card114')
from events import find, SLIDES
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
st=dict(np.load('runs/114/state_train.npz')); sv=dict(np.load('runs/114/state_val.npz'))
tr={k:v[:900*1001] for k,v in st.items()}; te_parts=[sv,{k:v[900*1001:] for k,v in st.items()}]
def feats(s,name,chunks):
    e=find(s)[name]; t=e['t']; a=s['actions']; ax=SLIDES[name][0]
    X=np.stack([a[t+5*c+j] for c in chunks for j in range(5)],1).reshape(len(t),-1)
    q=s[name]; d=e['dir']
    y=np.stack([(q[t+5*k]-q[t])*d for k in range(1,5)],1).max(1)>=0.04
    return X,y,e['locked']
for name in ['drawer','window']:
    for tag,chunks in [('past 2 chunks',[-2,-1]),('chunk 1 (t..t+4)',[0]),('chunks 1-2',[0,1]),('all 6 the model sees',[-2,-1,0,1,2,3])]:
        X,y,_=feats(tr,name,chunks)
        Xs,ys,ls=zip(*[feats(p,name,chunks) for p in te_parts]); Xt=np.concatenate(Xs); yt=np.concatenate(ys); lt=np.concatenate(ls)
        clf=GradientBoostingClassifier(random_state=0).fit(X,y); p=clf.predict(Xt)
        r=p==yt
        print(f'{name} {tag:22s} right on locked {r[lt].mean():.3f}  unlocked {r[~lt].mean():.3f}  balanced {(r[lt].mean()+r[~lt].mean())/2:.3f}')
