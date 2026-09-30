---
id: "036"
title: fresh codes for new things
rung: 0
serves: [P7, P5, P3, C5]
status: done
verdict: pass
arch_version: 5
date: 2026-09-29
---

# 036: fresh codes for new things

## 1. Question

Suppose each piece that card 035's rule marks "new" gets a fresh code in
its codebook, shared only by new pieces close to each other. Are the new
tiles then named apart from each other as well as from familiar ones?
Serves:
- P7, units worth tracking;
- P5, a sense of when a prediction applies;
- P3, a new colour as one new value, shared by the things that have it;
- C5, familiar codes unchanged.

This follows [card 035](../035-novelty-from-own-tiles/card.md), which
was stopped. Card 037, recall in the planner, needs every tile to have a
name of its own. Under card 035's reading every new piece shows the same
"new" marker, so new tiles collide. In 8 of 10 seeds at least two of the
8 new tiles had identical codes. In seed 301, the yellow key, both yellow
doors, the purple key and the purple doorway all had one tuple. CHARTER.md
already says "New codes for a new thing are fine."

## 2. What changes

One component: how a new piece is named.

| | Card 035 | Card 036 |
|---|---|---|
| A piece within α × its nearest code's radius | That code | That code |
| A piece beyond it | The one marker "new" | A fresh code of that codebook |

- **Fresh codes.** In each codebook, the pieces marked "new" are grouped
  by complete linkage: two groups merge only while every pair of pieces
  in them lies within α × m_k of each other. Here m_k is the codebook's
  median distance of training pieces to their codes, the radius floor of
  card 035. Each group gets a fresh code.
- **α = 6,** card 035's leave-one-out choice. Nothing new is tuned here.
- Familiar tiles never have a "new" piece (card 035), so their codes do
  not change.
- **Encoders:** card 035's saved μ = 0.01 encoders, seeds 300–309
  (`runs/035_enc/`). The reading is all that changes, so no retraining.
- **Declared exceptions:** as card 035. The planner is not run here; card
  037 does that.

**Arms:**
1. Upper bound: card 031's label codes (a new colour is a new value).
2. **Main: fresh codes.**
3. Baseline: card 035's single "new" marker.

## 3. Dependencies

- Card 035: the encoders, the novelty rule and α.
- Complete-linkage clustering with a distance threshold, a standard
  method.

## 4. Data check

28 tiles: the 20 familiar ones, plus a key, a closed door, an open door
and the agent in a doorway, each in yellow and in purple.

## 5. Feasibility gate

- **Upper bound:** arm 1 names all 28 tiles apart.
- **Trivial baseline:** arm 3 names them apart in 1 of 10 seeds (card
  035's reading, recomputed from its saved encoders).

Result of the gate (2026-09-29): passed. Arm 1 named all 28 tiles apart;
arm 3 did so in 1 of 10 seeds.

## 6. Success criteria and prediction

1. **Named apart.** In at least 8 of 10 seeds, all 28 tiles have
   different code tuples.

Also reported:
- how many fresh codes each codebook gets;
- which new tiles share a fresh code. In particular: do a colour's four
  new tiles share one fresh code in some codebook that the other colour's
  tiles do not? That is what "same colour" would later read (P3);
- the distance, relative to α × m_k, between new pieces that merge and
  pieces that do not.

**Prediction.** Criterion 1 passes. Pieces of different kinds (a key
against a door) lie farther apart than α × m_k, which is small because
training pieces sit close to their codes. Whether a colour's tiles share
a fresh code is open. A codebook that separates colour would show it,
and card 035's first probe saw every blue tile move together in one
codebook.

**Budget.** Seconds: no training, no planner.

## 7. Result

Numbers are in `results.json` and `runs/036_fresh.json`; the codes are
saved in `runs/036_codes/` for card 037. Approved by the user on
2026-09-29 and run the same day, in seconds.

| Criterion | Result | Verdict |
|---|---|---|
| 1. All 28 tiles named apart | 8 of 10 seeds (card 035's single marker: 1 of 10) | Pass |

- **The two failures were not fresh codes.** In seeds 307 and 309, the
  purple doorway (and in 309 the purple open door) was never "new". Card
  035's rule gave it the blue tile's codes in all four codebooks. Purple
  lies close to blue in these vectors.
- **A new colour is not one fresh code.** In no seed did a colour's four
  new tiles share a fresh code, and the key and closed door of a colour
  shared one in no seed. Almost every new piece got its own code: 60–110
  fresh codes for 75–117 new pieces over the 52 tiles the view can show.
  The cut, α × m_k, is small (about 0.01 in most codebooks), because
  training pieces sit close to their codes.
- **But what is shared shows.** The new open door and the agent standing
  in it shared their codes in three of the four codebooks in 16 of 20
  cases (colour and seed). They differed only in the codebook that
  carries the agent. So drawing the agent onto a new open door changes
  one codebook, the way card 031's per-codebook outcomes describe it.

## 8. Decision

**Keep** (2026-09-29; the user's plan was to run card 037 if this card
passed). Fresh codes are how card 037 names new pieces. Two limits carry
forward:
- a new colour is not yet one shared value, so "same colour" cannot be
  read from codes;
- the novelty rule can miss purple against blue.

## Appendix A: the procedure

`tools/card036/fresh.py` reads each saved encoder (pieces z, codebook
vectors), applies card 035's `read_codes` with α = 6, then for each
codebook k:
- takes the pieces marked "new" in k among the 52 tiles the egocentric
  view can show (every object, with and without the agent on it);
- clusters them by complete linkage, cut at distance α × m_k (a direct
  agglomerative implementation; scipy is not installed);
- gives cluster j the code M + j (8 + j) in codebook k. Its vector is the
  mean of its pieces, which card 037 uses for tiles that outcomes create.

Named apart is checked on the 28 tuples.
