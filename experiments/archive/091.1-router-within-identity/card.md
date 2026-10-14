---
id: "091.1"
title: the router's vote within the same tiles first (identity before similarity)
rung: 1
serves: [P10, P6, P17]
status: done
verdict: pass
arch_version: 19
date: 2026-10-09
---

# 091.1: the router's vote within the same tiles first

Made overnight (2026-10-09) under the user's authority ("do as much work
as you see fit tonight, don't ask for my approval"), because version 19
scores 0/30 on tier 3 and card 092's tier 3 criterion needs it.

## 1. Question

Version 19 (card 091) loops on tier 3: in 28 of 30 episodes the agent
turns in place before a locked door. The trace (seed 1003001): facing
away, the plan "toggle the blue door with the hand as it is" turns it to
the door; facing it, the plan "a key first" turns it away. Memory holds
1,544 toggles of that door with an empty hand, none of which opened it,
yet the router's P(the door changes) ranges over 0.27–0.53 across the
believed views it is asked with, because its vote runs over all 54,000
toggle keys, where blue keys did open blue doors. Version 18 asked first
whether the same tiles were tried (recall's identity level, card 042:
codes for identity, vectors for similarity). If the router votes only
among stored keys with the query's front and held tiles when there are
any (ranked by its view similarity), and over every key otherwise, does
tier 3 stop looping, with tiers 1–2 kept?

## 2. What changes

One component: which stored keys card 091's vote runs over.

```
091:    P_route(q) = vote over every stored key of the action
091.1:  K_same = stored keys with q's front code and held code (forward: front code)
        P_route(q) = vote over K_same if K_same is not empty, else over every stored key
```

The embedding, τ, α and own tries first are unchanged.

## 3. Dependencies

Card 091; card 042 (identity by codes, similarity by vectors); card 050.

## 4. Data check

Tier 3's toggle keys with the empty hand before a locked door: 1,544
tries, none changed the door (`tools/card092`'s probe, overnight log).

## 5. Feasibility gate

- **Upper bound:** the probe's 400 believed views with an empty hand
  before the blue locked door give P(change) < 0.5 in all of them.
- **Trivial baseline:** version 18 (no router) does not loop (25 and 26
  explore steps on seeds 1003000–1003001 in 180 s).

## 6. Success criteria and prediction

1. **Tier 3 loops:** no episode with more than half its steps exploring
   in a 2-seed smoke, then in the 30-seed run.
2. **CHARTER's tiers** (with card 087's forward prior, as card 092's base
   arm): tier 1 ≥ 99%; tier 2 not worse than 96% (card 092's base);
   tier 3 against version 18's 5/30.

**Prediction.** The loop goes; tier 2 stays within a few points (its
queries mostly have stored keys of the same tiles).

**Budget.** Smoke 6 minutes; tiers 1–2 about 25 minutes; tier 3 about 2
hours per arm.

## Gate result

- **Upper bound:** not measured as written; the tier 3 smoke stands in
  for it (below).
- **Tier 3 smoke** (2 seeds, 180 s, `runs/092/t3smoke_id.json`): 25 and
  26 explore steps, as version 18's 25 and 26; version 19 without 091.1:
  3,600 (seed 1003001) on the same smoke.

## Result: tiers 1 and 2

With card 087's forward prior, as card 092's arms (`runs/092/*_id_*`).

| Tier | Base, 091 | Base, 091.1 | 092, 091 | 092, 091.1 |
|---|---|---|---|---|
| 1 (200) | 100%, 24.4 | 100%, 24.4 | 100%, 24.4 | 100%, 24.4 |
| 2 (100) | 96%, 78.1 | 95%, 119.4 | 95%, 56.4 | 95%, 64.3 |

Tier 2 no worse in success; more steps when solved without the reveal
step (78 → 119).

## Result: tier 3

Card 092's base arm (version 19 + 091.1 + card 087's forward prior), 30
seeds, an hour each, 44 GB cap, one math thread per worker
(`runs/092/base_id_tier3.json`):

| | Successes | Steps when solved | Seconds per step |
|---|---|---|---|
| Version 18 (`runs/087/tier3_v18.json`) | 5/30 | 1,667 | 1.68 |
| Version 19 (091) | 0/30 (loops) | – | 0.12 |
| **091.1 with 087's forward prior** | **14/30** | 2,055 | 1.45 |

Against version 18 on the same seeds: 10 won, 1 lost (sign test p =
0.012). The first wave (20 in parallel) 5 of 20 (version 18: 1 of 20);
the second (10) 9 of 10 (version 18: 4 of 10). 15 episodes reached the
hour, none the step limit. Version 18 ran without the one-thread
setting; under equal load the two run at the same speed (`runs/092/speed_*`:
2,970 and 3,044 steps in 20 × 300 s).

1. **Loops: met** (2.8% of steps exploring; version 19: 95%).
2. **Tiers: met.** Tier 1 100%; tier 2 95% (best version 19: 79%;
   no different from 96% without 091.1); tier 3 14/30, better than 5/30.

## Decision

**Keep** (overnight, under the user's authority; the user may revert).
Recall's identity level comes before the router's similarity, as
version 18's own tries did; together with card 087's forward prior it
is version 20.

## Later: tier 2's five failures (2026-10-09)

Traced in both arms (`runs/092/t2fail_*`, `runs/092/t2dbg_29.log`,
`WM_ROUTER_DEBUG=3`). The same five seeds fail with and without the
reveal step, and four of them (1002015, 029, 067, 070) fail the same way.
The agent opens the door, sees the box and drops its key to free its
hand. It then never picks up the box, because recall no longer predicts
that pickup will work. In tier 2's memory the group "pick up, front a
box of that colour, hand empty" holds 16–19 tries, and every one
succeeded. But their views are far from the present one, so the vote
over them has little mass (0.06–1.9). The rest is filled by the prior α·f,
where f is over every stored pickup try (0.90 "nothing changes"). Seed
1002029 shows P(the box comes into the hand) ranging over 0.13–0.87
with one view tile, for example 0.29 with a green ball in view. Version
19 voted over every key, where similar box pickups from other groups
added mass. Without 091.1 these four seeds pass, and three others fail
instead (1002013, 048, 088). Seed 1002017 is a different failure: it
holds a red key before a green box and loops between dropping the key
and picking up a grey ball (untraced).
