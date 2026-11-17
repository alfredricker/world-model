"""Card 143, second data check: whose release point is a good label, and where a slide's hold ends.

Sets m400, m600, m800 (fit) and t500, t700 (score), from the state dataset; the truth only scores.
Cube: every touch across which the cube moved over 1 cm, split by where the cube is afterwards, by whether the touch
began on the cube (contact within 3 cm of it) and by whether the drawer moved too. Error of the sensed release point
against the cube's position before the next touch.
Slides: every touch across which a slide moved over 1 cm. The opening afterwards read by a linear fit from the hand's
position at three moments: the sensed release, the frame the slide stops (timing from the truth), and the frame before
the hand has risen 2 cm (proprioception only).

  bin/prun python tools/card143/check_hold.py > runs/143/check_hold.log
"""
import numpy as np
from pathlib import Path
ROOT=Path('.').resolve(); EP=1001
z=np.load(Path.home()/'.ogbench/data/scene-play-v0.npz')
def rows(first):
    sl=slice(first*EP,(first+100)*EP); obs,qpos=z['observations'][sl],z['qpos'][sl]
    name={400:'m400',600:'m600',800:'m800',500:'t500',700:'t700'}[first]
    c=np.load(ROOT/f'runs/122/computed_{name}.npz'); st,en=c['st'],c['en']
    out={s:[] for s in ('drawer','window')}
    cube=[]
    for i in range(len(st)-1):
        if st[i]//EP!=st[i+1]//EP: continue
        a,b=st[i],st[i+1]
        hand=obs[a:b,12:15]/10
        # the hand's speed; the end of the hold = first frame after contact where the hand rises 2 cm above its lowest height since contact, or the next touch
        zmin=np.minimum.accumulate(hand[:,2]); up=np.where(hand[:,2]>zmin+0.02)[0]
        e_up=(up[0]-1) if len(up) else len(hand)-1
        # stillest frame of the hand between the sensed release and the lift
        for s,j in (('drawer',23),('window',24)):
            q=np.abs(qpos[:,j]); 
            if abs(q[a-1]-q[b-1])>0.01:
                seg=q[a-1:b]; mv=np.where(np.abs(np.diff(seg))>1e-4)[0]
                stop=min(mv.max()+1,len(hand)-1)
                out[s].append((*hand[min(en[i]-a,len(hand)-1)],*hand[stop],*hand[e_up],q[b-1],q[a-1],*hand[2], stop-(en[i]-a), mv.min()))
        cb,ca=qpos[a-1,14:17],qpos[b-1,14:17]
        if np.linalg.norm(cb-ca)>0.01:
            cen=np.array([0.425,0,0]); h=hand+cen
            ins=lambda c:(abs(c[0]-0.33)<0.12)&(c[2]>0.04)&(c[1]<-0.1)
            dq=abs(qpos[b-1,23])
            wh='table' if not ins(ca) else ('open' if dq>=0.03 else 'hidden')
            oncube=np.linalg.norm(h[2]-cb)<0.03
            cube.append((wh,oncube,np.linalg.norm((h[min(en[i]-a,len(h)-1)]-ca)[:2]),np.linalg.norm((h[e_up]-ca)[:2]), abs(abs(qpos[a-1,23])-dq)>0.01))
    return {k:np.array(v) for k,v in out.items()},cube
L=[rows(f) for f in (400,600,800)]; T=[rows(f) for f in (500,700)]
cat=lambda X,s:np.concatenate([x[0][s] for x in X])
A=lambda x:np.c_[x,np.ones(len(x))]
for s in ('drawer','window'):
    l,t=cat(L,s),cat(T,s)
    print(s,'touches',len(l),len(t),'  sensed start after slide starts moving (frames) median',np.median(t[:,15]),' slide stops after sensed release (frames): median',np.median(t[:,14]),'90%',np.quantile(t[:,14],.9))
    # which touches are on the handle: contact point cluster. use where first motion is within the sensed touch (mv.min<=3)
    for sel_name,sel in (('all',lambda x:np.ones(len(x),bool)),('slide starts moving within 3 frames of contact',lambda x:x[:,15]<=3)):
        kl,kt=sel(l),sel(t)
        for nm,c0 in (('sensed release',0),('frame the slide stops (truth timing)',3),('hand lifts 2 cm',6)):
            B=np.linalg.lstsq(A(l[kl][:,c0:c0+3]),l[kl][:,9],rcond=None)[0]; e=np.abs(A(t[kt][:,c0:c0+3])@B-t[kt][:,9])*100
            print(f'   {sel_name:48s} n={kt.sum():5d} {nm:38s} median {np.median(e):5.2f} cm  within 1 cm {100*(e<1).mean():5.1f}  within 3 cm {100*(e<3).mean():5.1f}')
for nm,X in (('learn',L),('test',T)):
    C=[r for x in X for r in x[1]]
    for wh in ('table','open','hidden'):
        for on in (True,False):
            k=[r for r in C if r[0]==wh and r[1]==on]
            if k:
                e=np.array([r[2] for r in k])*100; e2=np.array([r[3] for r in k])*100
                print(nm,wh,'touch began on the cube' if on else 'touch began elsewhere','n',len(k),'release-point error median',round(np.median(e),2),'within 1 cm',round(100*(e<1).mean(),1),'| at hand-lift median',round(np.median(e2),2),'within1',round(100*(e2<1).mean(),1),'| drawer moved too',sum(r[4] for r in k))
