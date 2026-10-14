# Architecture version 10 (card 051): applied 2026-10-04

Staged here instead of in ARCHITECTURE.md, as promised to the user on
2026-10-04: the decisions below were Claude's under delegation, and the
user applies or rejects them. The user kept steps 1 and 4a–4d on
2026-10-04 and version 10 was applied. When applied, ARCHITECTURE.md gets
`arch_version: 10` and the change-log line at the end.

## What changes

Version 9 with five planner and memory changes from card 051:

1. **Memory indexed by situation** (step 1, `tools/card051/index.py`).
   - Recall for pick up, toggle and drop compares a query with groups of
     stored keys equal on the front tile, the held tile and the admitted
     view conditions, with summed outcome counts, instead of with every
     key. Exactly the same predictions.
   - A query's own situation is a hash lookup.
   - Admission (card 049) runs over per-candidate groups: while candidate
     c is weighed, keys equal on the front and held tiles, the admitted
     view conditions and c are one group. Exactly the same conditions,
     gains and α.
   - The planner's situations (card 047's templates and the situations
     that made a tile walkable) keep one situation per class (held tile,
     admitted view values).
   - Card 038's ways (keep, copy, shift or set) are fitted once on stored
     memory (at most 1,000 keys per class) and refitted with admission,
     not after every try.
2. **Threats between the needs of one achiever** (step 4a,
   `tools/card051/threats.py`). When an achiever has two or more unmet
   needs, each is planned from the present; each plan records what it
   relies on (needs met along its chain, and "this action works with the
   present hand and view" for achievers with no needs). A need whose
   plan's act, imagined on the present, breaks what another need's plan
   relies on is not pursued first.
3. **Commitment between steps** (step 4b, `tools/card051/commit.py`).
   Each step records the achiever it chose for every condition in its
   chain; the next step tries that achiever first and searches only when
   it no longer gives a plan.

4. **Routes through two tokens** (step 4c, `tools/card051/walk2.py`).
   When no route faces a target with one token made walkable, pairs of
   tokens that recall says can be made walkable are tried; the one the
   cheapest route steps onto first becomes the condition ("walk", j).
5. **The hand kept when only the view is asked for** (step 4d,
   `tools/card051/threats.py`, `HAND_LINK`). When card 043 asks only for
   the view because the hand works as it is now, the present hand is a
   protected link in that branch.

Results with all five (seeds 400–404): two doors 30/30 (version 9: 0/30),
one door 100% (98%), cluttered 100% except one layout (98–100%), the four
familiar worlds 100% with card 029's steps; one failure left in 1,950
test episodes.

## Sections of ARCHITECTURE.md to update

- **Intro:** version 10 = version 9 + card 051's index, threats and
  commitment; results: two doors 30/30 in every seed (version 9: 0/30),
  cluttered 99–100%, familiar worlds unchanged, per-step time flat with
  memory.
- **In brief, Recall:** "compared with groups of keys equal on what the
  admitted conditions read; own situation by hash".
- **In brief, Acting:** add threats between sibling needs and commitment
  between steps; remove "recomputed at every step" (it still recomputes,
  but tries the kept choices first).
- **In brief, timing:** 0.19–0.26 s per layout in the familiar worlds
  (version 8: 0.42–0.72); per step 5.7–9.9 ms from base memory to
  +20,000 stored keys.
- **Data structures:** add "Recall groups" (per action: group key →
  summed counts, representative key), "Own situations" (hash → counts),
  "Situation classes" (front tile → (held tile, admitted view values) →
  one stored situation), "Kept choices" (condition → achiever, per
  episode).
- **Known limits:**
  - replace "Cost grows with memory" with: per-step cost flat in memory
    in the tests (to +20,000 keys); re-admission on grown memory slower
    (50 s at +20,000 keys), run only on surprise; card 038's transport
    for new situations reads every key of an outcome class;
  - replace "No commitment between steps" with what remains: cluttered
    seed 403 layout 93 (an achiever whose needs alternate in form);
  - walking now considers pairs of blocked tokens, not more;
  - keep "unknown read as fails", "parts mix kind and colour",
    "conjunctions spliced".

## Change-log line

| 10 | 2026-10-04 | 051 | Memory indexed by situation (exact: the same decisions as version 9, per-step time flat from base memory to +20,000 stored keys, 5.7–9.9 ms against version 9's 43–4,640 ms); threats between the needs of one achiever; the chain's choices kept between steps; routes through two tokens made walkable; the hand kept when only the view is asked for. Two doors 30/30 in 5 of 5 seeds (version 9: 0/30), one door 100% (98%), cluttered 99–100%, familiar worlds 100% with card 029's steps. Kept by Claude overnight under the user's delegation; to confirm |
