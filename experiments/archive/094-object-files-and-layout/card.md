---
id: "094"
title: belief as object files and a layout, so deliberation reads only attended tokens
rung: 1
serves: [P17, P7, P15, P19]
status: done
verdict: fail
arch_version: 20
date: 2026-10-09
---

# 094: belief as object files and a layout

Drafted 2026-10-09 with the user. The user: belief should
be a narrower, attention-style memory, since most places are irrelevant
to any condition, and in Crafter the important tokens (an enemy) need
their out-of-view behaviour predicted; spatial understanding perhaps
inspired by the brain's "where" cells. The user set card 093 (the
intention held between steps) aside on 2026-10-09: its result would not
inform this card or the structure card (now 096), so this card is measured against version
20. Literature reviewed 2026-10-09 (section 3).

## 1. Question

Version 20's belief is a token per place of a lattice around the start
(638 places at 12 tiles; set per map since card 066): floor and walls
included, never faded. Holding it is cheap (placing and writing take
about 1% of a tier 3 step); reading it is not. The planner's candidates
are every distinct token in the map, recall reads the believed view, and
walking propagates over every placement. If belief is instead split the
way the brain splits "what is where":
- **object files:** a small set of attended tokens, each with what, where,
  when last seen and where it is expected now;
- **a layout:** per place, only walkable, blocked or never seen;

and deliberation reads only the object files while walking and
exploration read the layout, does attention keep every token the agent
needs, with no tier worse and no step slower? Rung 1; P17 (what is held
stays bounded as worlds grow), P7, P15, P19.

## 2. What changes

One component: what belief holds and which readers read which part.

```
version 20:  belief = token per lattice place (what), one placement (where)
             planner candidates = every distinct token in belief; recall's view = every token within 6 places
this card:   belief = object files + layout, one placement (unchanged)
             object file  = (token, where at last sight, step last seen, expected where now)
             layout       = per place: walkable | blocked | never seen   (from recall's forward kind, card 087's prior)
             planner candidates and recall's view = object files only;  walking and exploration = layout
```

- **Which tokens get an object file (attention):** the goal's token;
  tokens named in what version 20 keeps between steps (the planner's
  kept choices, card 060's door to look behind, card 072's current
  guess); and tokens recall says some action can change (pick
  up or toggle predicted to have an effect, card 087's properties where
  recall has no tries; cached per
  code). Floor and walls get none: they are the layout.
- **A file's identity is its token id** (its where at first sight, card
  044), not its what: two doors of one colour are two files, and a token
  seen again where its file expects it is the same file (object files'
  matching by expected where).
- **Capacity:** not limited in this card (tier 3 has tens of such tokens
  against hundreds of places). A limit of a few, as in human working
  memory, needs long-term spatial memory to hold the rest, which is a
  later card.
- **Expected where now:** the where at last sight, for every token, in
  this card: in MiniGrid nothing moves unless the agent moves it. Card
  096's structure memory will predict it per kind (an enemy moves toward
  the agent) with uncertainty growing since last seen.
- **Where, as in the brain's spatial cells:** the placement (one
  transformation composed from the learned moves) is path integration,
  the role grid and head-direction cells play; an object file's where is
  its vector from the agent, as object-vector cells code; the layout's
  blocked places from the agent are what boundary-vector cells code. The
  layout is allocentric (the lattice), read through the placement.
- **Unchanged:** the encoder, recall's stored keys and admission, the
  search (still run every step), walking's closeness propagation (now
  over the layout).
- **Declared exception:** stored tries keep the full believed view; only
  queries read object files. If admitted view conditions read floor or
  walls (section 4), stored views are re-read through the same filter.

## 3. Dependencies

Cards 044 and 061
(tokens, learned moves, placement); 067 (walking); 057 and 060
(exploration); 087 (properties as the layout's walkability where recall
has no tries); version 20's recall. Literature (reviewed 2026-10-09; LITERATURE.md's
current focus):
- **Object files** (Kahneman, Treisman and Gibbs 1992): features bound to
  a place-and-time index, matched by expected where. Here a file is
  indexed by its token id (its where at first sight), so two doors of
  one colour are two files.
- **Where as path integration** (the Tolman–Eichenbaum machine,
  Whittington et al. 2020; `2112_04035`): an action-specific
  transformation updates where, kept apart from what; memories bind what
  to where. The placement already is this; object files and the layout
  are memories indexed by it.
- **Vectors to objects and boundaries** (object-vector cells, Høydal et
  al. 2019; boundary-vector cells, Lever et al. 2009): an object file's
  where is its vector from the agent; the layout's blocked places are
  boundaries.
- **Attention over a set of object vectors as working memory**
  (`1911_07141`, working memory graphs, on BabyAI); **a 2D map memory**
  (`neural-map`, in mazes, position given). Evidence in worlds like ours
  for both halves.
- **Capacity** (Pylyshyn and Storm 1988; Cowan 2001): about 4–5 tracked
  objects. Not imposed here (section 2).

## 4. Data check

From version 20's traces (tier 2's 100 seeds; 5 tier 3 seeds for 300 s):
- per step, how many tokens deliberation reads now (planner candidates,
  recall queries) against how many object files there would be;
- which admitted view conditions read tokens that would get no object
  file (floor, wall);
- how often walking's walkability comes from a token that would be in
  the layout only.

**Result** (`tools/card094/datacheck.py`, `runs/094/datacheck_*`; version
20; attention as revised below):

| | Tier 1 (50) | Tier 2 (100) | Tier 3 (2 × 300 s) |
|---|---|---|---|
| Planner candidates (distinct tokens) → attended | 3.7 → 1.7 | 4.5 → 2.7 | 14 → 12 |
| Believed places → object-file places | 45 → 1.7 | 44 → 10 | 156 → 30 |
| Recall's view set → attended part | 3.7 → 1.7 | 4.5 → 2.5 | 11 → 9 |
| Tokens acted on in solved episodes, unattended | 0 of 100 | 0 | – (none solved in 300 s) |

- The only unattended tokens in any stored view are floor and walls,
  and every stored view holds both.
- Recall's admitted view conditions read only keys, balls and doors,
  all attended.
- The only unattended tokens ever acted on are floor tiles, by random
  actions in failed episodes.

**Attention revised before the gate.** The first rule left out a token
when memory held tries on its code that never changed it. That left out
exactly what card 072's guesses exist for, such as a locked door of a
colour never opened in memory. It contradicts "a failed try is
situational", so a token now gets no file only when memory has never
seen it change *and* card 087's properties say no action changes it.

## 5. Feasibility gate

- **Upper bound:** with object files chosen by hand (every door, key,
  ball and box; oracle attention), deliberation's time per step on tier 3
  and tier 2's success. If hand-chosen object files lose tier 2 seeds,
  the planner needs more than they hold and the card stops.
- **Trivial baseline:** version 20 (belief as now).

**Gate result: fails as written** (`runs/094/oracle_tier2.json`). With
object files chosen by hand, tier 2 is 86/100 against version 20's 95:
3 seeds won, 12 lost (p = 0.035). The attention rule gives the same 86,
seed for seed (`runs/094/1_tier2.json`). Tier 1 is unchanged (200/200,
24.4 steps).

**Where the loss is** (tier 2, 100 seeds, `runs/094/1_*_tier2.json`).
Two readers changed: the planner's candidates, and recall's view as the
router reads it. The router was retrained on object-file views
(`runs/094/krouter_files.pt`; held-out likelihood as good as or better
than card 091's), since its network had only seen views with floor and
walls.

| Arm | Tier 2 | vs version 20 |
|---|---|---|
| Planner candidates from object files, version 20's router | **95** | identical: the same outcome and steps on all 100 seeds |
| Retrained router, full views | 92 | 2 won, 5 lost |
| Retrained router, full views, + planner candidates from object files | 92 | the same seeds as the row above |
| Retrained router reading object-file views | 86 | 3 won, 12 lost |

- **The planner needs nothing the object files leave out.** Removing
  floor and walls from its candidates changes no decision on tier 2.
- **Recall loses no information without floor and walls.** Its view is
  a set of distinct tokens with positions dropped, and floor and walls
  are in every stored view and every query (data check). So their
  removal tells recall nothing new; the 12 lost seeds are the retrained
  router's response to smaller sets. Retraining alone, with full views,
  already moved 7 seeds (2 won, 5 lost).
- The 3 seeds won (1002015, 017, 067) are three of version 20's five
  remaining failures, where a box pickup is voted down by stored tries
  whose views differ somewhere in view (card 091.1's note). What recall
  lacks is not floor and walls but *where* each token is: a key across
  the room should not make a box pickup look unlike.
- **Two side findings:**
  - On two seeds version 20 has no plan, and its only guesses are
    "toggle a wall". Carrying them out moves the agent until it sees
    what it needs. Restricting card 072's guesses to object files
    removes that movement and loses both seeds. The guesses read every
    seen token, as before (`WM_FILES_PARTS`).
  - Tracing (`WM_TRACE=1`) changes version 20's choices on some seeds.
    Without it, episodes are identical across hash seeds.

## 6. Success criteria and prediction

1. **Attention:** on every tier, the object files hold every token the
   solved episodes acted on (the goal's, and each picked up, toggled or
   dropped), and no tier 2 or 3 seed is lost for want of one.
2. **CHARTER's tiers:** tier 1 ≥ 99%; tiers 2 and 3 not worse than the
   best version (20: 95%, 14/30).
3. **Cost:** time per step on tiers 2 and 3 not worse than version 20's,
   with belief's size (object files and layout places) reported.

**Prediction.** The planner's candidates are distinct tokens, about 30 in
tier 3, so object files remove few of them (floor and walls); most of a
tier 3 step is the fallback's repeated guess search, which this card
does not touch. So tier 3 is not expected to run much faster. The card's
value is the form of belief Crafter needs (attended tokens with when and
where last seen, a layout for walking) at no loss on the tiers. The risk
is a condition on a token class attention leaves out (an admitted "wall
in view").

**Budget.** Data check and gate: about 30 minutes. Tiers 1–2: about 25
minutes. Tier 3: about 2 hours, handed to the user.

## 7. Result

The main runs were not made: the gate failed (section 5). Tier 1 200/200
(24.4 steps) and tier 2 95/100 with the planner on object files,
identical to version 20; 86/100 with recall's view also on object files.

1. **Attention:** met where measured. Every token acted on in solved
   episodes had a file (tiers 1–2), and the planner lost nothing.
2. **Tiers:** tier 1 met; tier 2 met for the planner alone (95,
   identical), not met with recall on object files (86); tier 3 not run.
3. **Cost:** not measured under equal load. Belief's reads: planner
   candidates 4.6 → 4.0 on tier 2, 14 → 12 on tier 3; object-file places
   10 of 44 (tier 2) and 30 of 156 (tier 3).

## 8. Decision

**Revise** (with the user, 2026-10-09). The planner needs no more than
object files hold, and walking reads the layout, but in MiniGrid that
changes no decision. Recall's view gains nothing from floor and walls in
set form; what it lacks is each token's position. The user: the spatial
element of recall is objects with their position relative to the agent,
not the layout, which is for moving. That revision is
[card 094.1](../094.1-relative-position-recall/card.md).
