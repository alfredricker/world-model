---
id: "085.3"
title: recall's weight fits on sampled query keys, with version 18's constant rate
rung: 1
serves: [P17, P18]
status: done
verdict: fail
arch_version: 18
date: 2026-10-08
---

# 085.3: sampled query keys, constant rate

Card 085.2's revision, approved by the user (2026-10-08: "sure thing").

## 1. Question

Card 085.2 sampled only the query keys of recall's weight fits (64 per
step, each compared with every stored key), which is unbiased and
exact when nothing is sampled. Its gate failed only through the
learning rate it carried over from card 085.1, which falls to zero over
the last half: version 18's fit has not converged after its 1,500
steps, so that rate stops 3–6% short. With version 18's constant rate,
does it pass the gate, build tier 3's memory within 20 minutes, and
grow linearly? P17, P18.

## 2. What changes

One thing against card 085.2: the learning rate is version 18's,
constant (`WM_SAMPLE_DECAY=0`). Everything else is card 085.2's
(`WM_SAMPLED=1 WM_SAMPLE_QUERIES=1 WM_SAMPLE_BQ=64`,
`tools/card085/sampled.py`). At 2,048 keys or fewer the exact fits run
(tiers 1 and 2 identical, card 085's gate part A).

## 3. Dependencies

Card 085.2 (the query-sampled path, shown exact with every key at every
step; its diagnosis `runs/085.2/diag_q64_const*` is this card's
configuration); card 085 (the exact reference, `measure.py`).

## 4. Data check

As card 085: the 1.5% cut for the gate; tier 3's full memory (54,119
keys and 26,050 distinct sets for the largest action) for the criteria.

## 5. Feasibility gate

Card 085.2's gate, unchanged (category ≥ 99%; each likelihood within 2%
or 0.001 nats per key of the exact weights'). Card 085.2's diagnosis
ran this configuration and passed it; the gate is rerun as declared.

## 6. Success criteria and prediction

Card 085's criteria:
1. **Tier 3 builds:** setup within 20 minutes, under 24 GB of RAM and
   16 GB of GPU memory.
2. **Linear:** setup time and peak RAM on 1/8, 1/4, 1/2 and all of tier
   3's memory grow at most linearly (log–log slope at most 1.2).
3. **No change where it worked:** tiers 1 and 2 identical (card 085).

**Prediction.** The gate passes as in the diagnosis. Each step compares
64 view sets with 26,050, so the fits take about 10 minutes at full
size. Risk: a part of setup never reached before grows with the square
of memory; if so it is reported with its sizes and becomes its own
card.

**Decision rules.** Keep if the gate and 1–3 hold (next: version 19 on
all three tiers); revise if the gate and 1 hold and 2 does not; stop if
the gate fails.

**Budget.** Gate about 1 minute. Criteria: four setups one at a time
(all, 1/8, 1/4, 1/2), each under a 20-minute limit and a 24 GB cap,
about 50 minutes.

## 7. Result

`runs/085.3/` (`gate.sh`, `scale.sh`, `rerun_quarter.sh`).

**Gate: passed.** Leave-one-out log-likelihood per key against the exact
weights: card 039's λ within 0.9%, 1.7%, 1.5% (pick up, drop, toggle);
card 042's within 0.0006 nats (pick up), 1.8% (drop), 0.5% (toggle);
category agrees on 99.8%, 100%, 99.9% of keys.

**Deviation: a GPU fault, fixed.** At 1/4 of the memory the first run
stopped after 37 s and its rerun after 296 s, each on an NVIDIA fault
(Xid 13, then Xid 31: an out-of-bounds write). The fused weighted-L1
kernel indexes (query things × stored things × 32 numbers) places, which
passed 2³¹ once view sets grew; the slices of stored sets are now sized
to keep it below 2³⁰ (`match_feats`). The second rerun completed. The
full and 1/2 runs reached the 20-minute limit without a fault, before
the fix.

| Tier 3's memory | Largest action: keys, sets | Setup | Peak RAM | Peak GPU |
|---|---|---|---|---|
| 1.5% (gate) | 839, 422 | 54 s | 2.8 GB | 0.4 GB |
| 1/8 | 7,172, 4,070 | 498 s | 3.0 GB | 0.7 GB |
| 1/4 (after the fix) | 14,325, 8,014 | 1,019 s | 3.3 GB | 1.0 GB |
| 1/2 | | over 1,200 s (limit) | | |
| all | 54,119, 26,050 | over 1,200 s (limit) | | |

- **Criterion 1: not met.** Setup at full memory did not finish in 20
  minutes; at the measured rate it would take about 68 minutes.
- **Criterion 2: met on the sizes measured.** Time from 1.5% to 1/8 to
  1/4: log–log slopes 1.05 and 1.03; RAM nearly flat (2.8 to 3.3 GB).
  The 1/2 and full runs stopped at the limit, as the linear rate
  predicts (about 34 and 68 minutes).
- **Criterion 3: met** (the exact path below 2,048 keys is untouched;
  card 085's part A).
- **Where the time goes** (1/4): the fits are 990 of 1,019 s. Card
  042's two-level fit takes 215–264 s per action, card 039's λ 85–102
  s, α 3–4 s. Each step compares 64 query sets with every stored set,
  so a step costs time linear in memory, over a fixed 1,500 or 2,000
  steps.
- **What version 18 reads from these fits.** Card 039's λ: recall's pair
  distance (card 074.2's cache) and card 038's ways (`key_dists`), so
  view weights included. Card 042's fit: only the front part of λ2 (the
  planner's situations, card 047); its β was replaced by card 050's, and
  its view weights by card 049's admitted conditions in every
  prediction. About 70% of setup fits weights version 18 no longer
  reads.

## 8. Decision

**Revise.** The fit is unbiased, matches the exact fit within the gate,
and grows linearly with memory, so tier 3's memory can be built; it
takes about an hour, not 20 minutes. Two revisions follow, for the
user: (a) save the fitted weights per memory, so the hour is paid once
and version 19's trial on tier 3 can go ahead (no change to the agent);
(b) fit card 042's level only for what version 18 reads (the front
part), checked on tiers 1 and 2, which should bring tier 3's setup near
20 minutes.

**Revision (a), done** (the user, 2026-10-08: "store the weights"; no change
to the agent). `tools/card085/stored.py` (`WM_STORED_FITS=1`) keeps each fit
in `runs/fits/`, keyed by a hash of everything it reads: the stored keys' and
view sets' vectors, the outcome counts, its settings, the sampling flags and
the fitting code. A new encoder, a different memory or changed fitting code
fits again. Check (`runs/085.3/stored_check.sh`): every fitted weight
bit-identical without storing, when stored and when loaded (tier 1: 23 of 23;
tier 2: 23 of 23); setup 25 s → 14 s on tier 1, 143 s → 24 s on tier 2.
Tier 3's one full build (about 68 minutes) is handed to the user.
Tier 3's build (`runs/085.3/tier3_build.sh`, run by Claude at the user's
request, 2026-10-08): the first attempt stopped on a GPU fault (Xid 13)
after the first action's three fits (pick up: λ 324 s, α 12 s, card 042's
level 1,295 s), which were stored; the rerun loaded them and fitted the
other two actions (toggle 1,412 s, drop 1,724 s), setup 3,259 s, peak RAM
5.7 GB, GPU 2.5 GB (`runs/085.3/tier3_stored.json`). Tier 3's memory now
loads from `runs/fits/`.

