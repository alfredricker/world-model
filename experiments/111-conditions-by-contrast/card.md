---
id: "111"
title: conditions by contrast — what remembered successes share and failed tries lack
rung: 1
serves: [P12, P10, P19]
status: done
verdict: uninterpretable
arch_version: 27
date: 2026-10-17
---

# 111: conditions by contrast over memory

Roadmap step 2. Drafted at the user's request, 2026-10-17. The user set the
sources of conditions as deduction from memory, experiment and others
(CHARTER, "Where statements come from"); imagined alteration is no longer
one. World 1 and Crafter are development worlds. The Craftium follow-up is
111.C, once Craftium's adapter gives tokens.

## 1. Question

Can the conditions of one-step effects be deduced from memory alone,
before any experiment? They are what the remembered successes of an effect
share and the failed tries lack, found by contrasts over real experience
only. Do they reach rung 1's bar (precision and recall ≥ 0.9 for every
kind of effect), including a lamp going out? Imagined alteration got 0.67
and 0.40 there (106, 110). Rung 1. Serves P12 (conditions), P10 (memory
answers what the view cannot) and P19.

## 2. What changes

One component: the source of conditions. Imagined alteration (105.4, with
106's filter) is replaced by contrast over memory. Neither System 1 nor the
interventions are used.

```
106:   condition of e = token j whose replacement in System 1's input changes the predicted outcome
111:   memory = every stored step: tokens' codes and places, the action, each token's next code
       event e: token i with code c0, action a, outcome c1 (i's next code)
       Q = remembered steps with action a and a token of code c0 at the anchor; success: outcome c1
       facts of e: for each token j and each anchor (i, and the agent's own token), (anchor, offset, code = c)
               for j's code, and (anchor, offset, code ≠ c') for every other code c' seen there
       greedy, as conditional contrasts (Cheng and Novick 1990): C = {}; repeat over Q restricted to C:
           add the fact with the largest causal power (Cheng 1997) if both sides hold ≥ 5 steps and a
           one-sided Fisher test gives p < 0.01 / (number of facts); stop when none qualifies
       conditions of e = the tokens j whose facts are in C
```

- **Changed after the smoke test (2026-10-17), before the main run.** The
  greedy step picks the fact with the smallest Fisher p, with ties broken
  by power. Picking the largest power picked small subsets that were pure
  by chance: for a blocked move, a wall 4 rows up and 6 across, power 1.0.
- **A fact can be a negation.** "No box on the lamp's other side" is a fact
  about a tile that shows floor.
- **Contrast inside the facts kept so far.** This separates a cause from
  facts that come with it.
- **Absent facts.** A fact that never varies in memory is not found. That
  case is for experiment, a later card.
- **Codes give identity** (the user, 2026-10-01). Similarity by vectors
  (wood 3 like wood 5) is a later card.
- **No tallies are stored** (recall replaces counting). Contrasts are
  computed from the stored steps when an event is asked about.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Facts are tokens at offsets from an anchor | yes (world frame) | yes | yes | yes, offsets binned | declared; vector similarity over offsets is a later card |
| An anchor is the effect's token or the agent's own token | yes (pose) | yes | effect's token only where there is no body | yes (odometry) | general; the body is proprioception (C1) |
| Facts are perception's codes, crisp | yes | yes | yes | coarse | declared; vectors for similarity later |
| Failures are remembered tries with the same action | yes | yes | yes | continuous actions need a kernel | declared, as card 110 |
| One-step effects | | | | | rung 3 (skills) takes effects over many steps |

## 3. Dependencies

- **Events and truth:** 105.4's events and its truth (the world's rule
  under one replacement), the same draw as cards 106 and 110
  (`tools/card110/`). All events are asked; none is dropped for a model's
  error, since no model is read.
- **Memory:** the training sets of world 1 (300,000 steps, card 102) and
  Crafter (132,785 transitions, card 105.10's placing).
- **Literature** (added when approved):
  - Cheng and Novick 1990, contrasts within focal sets;
  - Cheng 1997, causal power;
  - Pasula, Zettlemoyer and Kaelbling 2007 (listed): rules over the
    objects an action touches, learned from experience;
  - Winston 1970, near misses.

## 4. Data check

**Remembered occurrences in training:**
- world 1: lamps lighting 560, going out 395, pushes 6,545, moves 178,596,
  blocked moves 52,325;
- Crafter: tools made 987, blocked moves 4,373, moves 27,448.

Two boxes beside one lamp are rare in training (105.4). The condition
"no box on the other side" may lack the 5 steps a side it needs. That is
the most likely miss.

## 5. Feasibility gate

- **Upper bound:** the same contrasts over the evaluator's kinds and
  counts in place of perception's codes (scoring only).
  - Pass: precision and recall ≥ 0.9 on every kind, in both worlds.
  - If a kind misses, the method or the data is short, and the card says
    which by checking the counts each side.
- **Trivial baselines:**
  - "every token near the change" (105.4: precision 0.09–0.19);
  - imagined alteration (106's K: lamp going out 0.67 / 0.40).

## 6. Success criteria and prediction

Run on perception's codes. No model is trained or read, so there is one
run per world.

1. **Rung 1's bar.**
   - Precision and recall are each at least 0.9 for every kind of effect
     with at least 100 held-out events: world 1's five, and Crafter's
     makes, blocked moves and moves that succeeded.
   - Each is above the baseline "every token near the change".
2. **No loss against imagined alteration.** Where 106's K passed, recall
   and precision stay within 0.05 of it. Those kinds are world 1's lamps
   lighting, pushes, moves and blocked moves, and Crafter's three kinds.
3. **Few examples (P19).** With memory cut to 100 remembered occurrences
   of the kind, precision is at least 0.9 and recall at least 0.8, for
   every kind. The full curve (10, 30, 100, 300) is reported.

**Reported:**
- the facts kept per kind (the statements);
- events whose conditions were exactly right;
- the misses, by fact and by count each side;
- the cost per event against the memory's size (P17).

**Prediction:**
- **World 1's moves, pushes and blocked moves pass.** Tens of thousands of
  tries vary every neighbour.
- **A lamp lighting passes.**
- **A lamp going out** reaches precision near 1.0, since only real states
  are compared. Its recall stays near 0.6: the second box is too rare,
  which leaves that condition for experiment.
- **Crafter's makes pass** on the table and the materials.
- **Blocked moves:** creatures as blockers need enough cases.
- **Criterion 3** costs recall on the rarest facts.

**Decision rules:**
- **Keep** (contrast over memory becomes step 2) if 1–3 hold, or if they
  hold except where the count of a needed fact is below 5 a side. Such a
  miss is the experiment card's to close, and the card names it.
- **Revise** if a kind fails with enough cases each side. The method is
  then short: facts that need disjunction, or similarity.
- **Stop** if the upper bound fails on kinds with enough cases. Then
  contrasts over memory cannot state these conditions.

**Budget:** no training.
- Memory is coded once per kind as one-hot facts (world 1 about 0.6
  million anchored steps by 1,500 facts; Crafter about 30,000 by 4,300),
  on the GPU.
- Each event costs a few matrix products per greedy step.
- About 5 minutes per world, capped, after a smoke test of 30 events per
  kind.

## 7. Result

Run 2026-10-17 ([results.json](results.json); `tools/card111/`). Every run
was capped and took 13–70 s. A greedy step costs 3–11 ms per event.
- **World 1:** its codes map one-to-one onto the evaluator's kinds, so
  codes and labels give the same numbers.
- **Events:** 106's draw in both worlds.
- **Baseline:** "near the change" has precision 0.04–0.38.

| Kind (events) | Labels: precision / recall (the gate) | Codes: precision / recall | 106's imagined alteration (K) | P19 at 100 occurrences (codes) |
|---|---|---|---|---|
| World 1, lamp lights (213) | 0.85 / **0.55** | same | 1.0 / 1.0 | 0.96 / 0.27 |
| World 1, lamp goes out (141) | **0.71 / 0.49** | same | 0.67 / 0.40 | 0.67 / 0.27 |
| World 1, push (300) | 1.0 / 1.0 | same | 1.0 / 1.0 | 1.0 / 0.61 |
| World 1, move (300) | 1.0 / 1.0 | same | 1.0 / 1.0 | 1.0 / 0.23 |
| World 1, blocked (300) | **0.35 / 0.27** | same | 1.0 / 1.0 | 1.0 / 0.91 |
| Crafter, make (236) | 0.97 / 0.92 | 0.97 / 0.92 | 0.99 / 1.0 | 0.99 / 0.51 |
| Crafter, blocked (300) | 1.0 / 1.0 | **0.86 / 0.89** | 1.0 / 0.95 | 1.0 / 0.89 |
| Crafter, moved (300) | 0.97 / 1.0 | **0.54** / 1.0 | 1.0 / 1.0 | 0 / 0 (too few failures) |

**Gate: fail in world 1, pass in Crafter.** On the evaluator's labels,
Crafter meets rung 1's bar on all three kinds. World 1 misses it on lamps
and blocked moves, with hundreds of cases of each.

1. **Rung 1's bar: fail.** It passes for world 1's pushes and moves and
   Crafter's makes. It fails for world 1's lamps and blocked moves, and
   for Crafter's moves on perception's codes.
2. **No loss against imagined alteration: fail.** World 1's blocked moves
   fall from 1.0 to 0.35 / 0.27, and Crafter's moves to 0.54 precision. A
   lamp going out improves (0.71 / 0.49 against 0.67 / 0.40).
3. **P19 at 100 occurrences: fail** (recall 0.23–0.91).

**Why, from the diagnostics** (each missed true condition is classified in
the final subset of memory; each false one by what it is):
- **Position proxies (world 1's blocked moves and lamps lighting).**
  - In a world with fixed borders, a far cell's kind is a stand-in for
    the agent's place: "a wall 5 rows up" holds exactly when the agent is
    in the top rows.
  - With 59,000 remembered tries, such a proxy is often the most
    significant fact. It makes the subset pure, and the true cell is
    never contrasted: all 239 of the blocked moves' misses are "no
    contrast left".
  - With memory cut to 100 occurrences, the proxies lose significance,
    and blocked moves reach 1.0 / 0.91.
  - Memory alone cannot tell a proxy from a cause when the two always
    occur together. That is experiment's job.
- **Rare variation.** Most of the misses for lamps (222 and 140) are
  cells that never vary in the remembered subset, such as a second box
  beside a lit lamp. This is also experiment's case.
- **The truth's convention (a lamp going out).**
  - Every false condition, 141 of 141, is the box being pushed away.
  - 105.4's truth does not count it. Replacing it gives a lit lamp with
    no box, which the world's rule recomputes as unlit, so "nothing
    changes".
  - In plain terms the box is a condition, so precision on real
    conditions is 1.0.
- **"Enough cases each side" misses** (lamps 83 and 136, makes 36,
  Crafter's blocked moves on codes 32). These are cases where the method
  itself is short. A greedy path takes one branch of a disjunction (one
  table of two, for example), and a code split can make the remaining
  cases look different.
- **Crafter's moves on codes.** 185 of the false conditions are the
  health slot, full. Losing health comes with a zombie beside the agent,
  which also blocks. It is the same proxy pattern, and on labels it does
  not win.

**Correction (2026-10-18, card 111.1's data check).** The "position
proxies" reading of world 1's blocked moves was mostly a numerical fault.
Fisher p-values below 1e-300 were clamped there, so with 59,000 cases
every strong fact tied and column order chose among them. With p kept in
log space, blocked moves reach 0.96 / 1.0. That is also why cutting memory
to 100 occurrences helped. Lamps and Crafter do not change. The decision
below stands.

## 8. Decision

**Keep** (the user, 2026-10-17), replacing a first decision of stop.
- **Why the first decision was wrong.** The bar asked memory alone for
  every condition. CHARTER gives conditions three sources: memory,
  experiment and others. The card's own keep rule already handed facts
  with fewer than 5 cases a side to experiment. The same holds for facts
  that always occurred together. Memory cannot separate them, by
  construction. So the card cannot fail on them, and the verdict
  against rung 1's bar is uninterpretable. That bar is judged once
  experiment and demonstration join memory.
- **Where memory varies the facts independently, it works:** Crafter on
  labels (0.97 / 0.92, 1.0 / 1.0, 0.97 / 1.0), and world 1's pushes and
  moves (1.0 / 1.0).
- **What stays memory's to fix** (not experiment's):
  - **Ties.** Where facts held on exactly the same remembered cases,
    memory picked one by column order. It should return them as one tied
    set, which tells experiment what to separate. (Blocked moves'
    precision of 0.35 was mostly a numerical fault; see the correction
    above.)
  - **Misses with enough cases each side** (lamps 83 and 136, makes 36,
    Crafter's blocked moves on codes 32). A greedy path takes one branch
    of a disjunction.
- **Next: experiment.** The agent changes one candidate fact in the world,
  chosen where memory returned a confounded set or fewer than 5 cases a
  side, and the result enters memory. Rung 1's bar is judged on memory
  and experiment together.
