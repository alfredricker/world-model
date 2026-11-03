---
id: "106"
title: a sleep phase — conditions read from remembered events, only in states the world could be in
rung: 1
serves: [P10, P12, P4, P19]
status: done
verdict: revise
arch_version: 27
date: 2026-10-13
---

# 106: a sleep phase

Approved by the user 2026-10-13 ("run 106").
Drafted at the user's request, 2026-10-13: "going through memory or recall
and understanding the conditions that made things true is good during a
sleep phase". The first draft's loop placed the view with the model's
predicted next tile. The user rejected it as not portable ("a sense of
'where' … should be fed through the perception"). The pose now places the
view (GOAL C1 as amended; card
[105.10](../105.10-proprioception/card.md)), so sleep does the brain's
work only.

## 1. Question

**Where things stand.**
- **Conditions are read by intervention** (105.4). A token is replaced by
  another kind's vector, and the model's predicted effect is checked.
  These conditions are exact for world 1's pushes and moves.
- **Some of them are false.** 493 of the 501 false conditions of a lamp
  going out came from replacements that make an impossible state: a lit
  lamp with no box beside it. The user's rule of 2026-10-01 says to check
  conditions only in situations the world's learned effects produce.
- **They are read once, offline, and not stored.** 105.5 learned its
  statements from 6,000 training steps before acting.

**The question.** Suppose that between episodes the agent sleeps. It goes
through its remembered events: every stored step where the model predicts
a token changes. It reads each event's conditions by intervention, and
keeps a replacement only if the world could be in the resulting state. The
conditions are stored as statements (105.5), merged across events.

Are the stored statements then right in both worlds? Specifically, is
precision at least 0.9 for every kind of event, including a lamp going out
(0.66 now), with nothing lost elsewhere? Rung 1. Serves P10 (consolidate),
P12 (conditions) and P4.

## 2. What changes

One component: which interventions count. The interventions themselves are
105.4's: one token replaced at a time, and the model's predicted effect
compared.

```
105.4:      every replacement counts; token j is a condition if some replacement changes the predicted effect
arm H:      a replaced state counts only if the world could be in it: under the model's own prediction for
            doing nothing (no-op), the event's token and the replaced token keep their codes
            (a lit lamp with no box beside it is predicted to go out, so that state is set aside)
arm K:      a replaced state counts only if the model knows it: two independently trained copies predict
            the same outcome for it (LESSONS: query a network only where copies agree)
arm HK:     both
sleep:      run over the remembered events after an episode; statements (action, effect, conditions as
            offsets and codes) merged by count, as 105.5, and kept in memory for the planner
```

**Why two filters.** They catch two different failures:
- **H, states the world cannot be in.** 493 of 501 false conditions of a
  lamp going out (105.4) came from these. A replacement the model would
  immediately undo shows a state the world never sits in.
- **K, states the model has no experience of.** All of Crafter's false
  make conditions (105.10) came from these: a plant (99 flips), a zombie
  (54) or a table (44) placed 2–4 tiles from the agent. Such states are
  possible, but almost absent from the curriculum's daytime crafting.
- **Portability.** Both use only the brain's own model and memory, so
  both carry to every world.

**The models:**
- world 1: 105.6's, seeds 1021, 7 and 3 (pairs of seeds for K);
- Crafter: 105.10's, seeds 1021 and 7.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| The agent can pause to sleep | yes | yes (between levels) | yes (between sessions) | general (brain) |
| Doing nothing is an action | yes (no-op) | not in every game | yes (zero command) | declared; where absent, the shortest wait |
| A state the world sits in is held by its own dynamics | mostly (arrows and creatures move by themselves) | yes | mostly | general; moving tokens are judged on their most likely next code |

## 3. Dependencies

- **105.4's interventions** (`tools/card105.4/world1.py`, `crafter.py`)
  and their scoring against the world's rule.
- **105.5's statements and planner** (`tools/card105.5/world1.py`).
- **Models:** 105.6's (world 1) and 105.10's (Crafter).
- **Literature** (added when approved):
  - Pearl 2009 and Woodward 2003: an intervention must stay within the
    system's possible states;
  - McClelland, McNaughton and O'Reilly 1995: replay during sleep
    consolidates.

## 4. Data check

The events are 105.4's held-out events, unchanged:
- **World 1:** pushes, moves, blocked moves, and lamps lighting and going
  out.
- **Crafter:** 300 blocked moves, 300 moves that succeeded, and 232 makes
  that made a tool.
- **What H and K can change:** they only remove replacements, so they can
  raise precision but not recall. World 1's missed second box (lamp going
  out, recall 0.40) is not this card's target. It needs the agent's own
  tries (waking).

## 5. Feasibility gate

- **Upper bound:** keep only replaced states that the world's own rule
  calls possible and that appear in training memory (scoring only).
- **Trivial baseline:** 105.4 and 105.10 unchanged:
  - lamp going out: precision 0.66;
  - Crafter makes: precision 0.77 and 0.84.
- **Gate, arm H in world 1:**
  - it keeps at least 95% of the replacements the world's rule calls
    possible;
  - it sets aside at least 90% of those it calls impossible.
  - If not, arm H is dropped and K runs alone.

## 6. Success criteria and prediction

For the strongest arm:
1. **World 1** (three seeds).
   - Precision is at least 0.9 for every kind of event, lamps going out
     included.
   - Recall elsewhere is no lower than 105.4's (moves and pushes 1.0;
     lamps lighting 1.0 in two seeds of three).
2. **Crafter** (two seeds).
   - Makes reach precision of at least 0.9 with recall of at least 0.95.
   - Moves keep precision and recall of at least 0.9.
3. **Planning.** 105.5's planner, run on statements stored by the sleep,
   lights the lamp in at least 84.5% of solvable world-1 test worlds.

**Prediction:**
- **H** fixes world 1's lamp going out (0.66 → about 0.95) and does
  little in Crafter.
- **K** fixes Crafter's makes (about 0.95) and costs a little recall
  where copies disagree on rare true conditions.
- **HK** passes 1 and 2.
- **Criterion 3** holds, because the planner's statements were already
  right.

**Decision rules:**
- **Keep** (sleep reads conditions through both filters) if 1–3 pass.
- **Revise** if one world passes.
- **Stop** if neither filter raises precision by at least 0.1 where it
  is aimed.

**Budget:** interventions and scoring, all arms: about 15 minutes in short
capped runs.

## 7. Result

Run 2026-10-13 ([results.json](results.json); `tools/card106/world1.py`,
`crafter.py`, `plan_world1.py`). Each run took under a minute.

**Gate: H fails.** In world 1, H set aside 0 of 39,744 replaced states
that the world's rule calls impossible (bar: 90%). It kept all 206,835
possible ones. The result is the same in all three seeds.
- **Why:** in world 1 doing nothing never changes anything, because every
  real state is consistent. The model learned that a no-op leaves
  everything as it is, impossible lit lamps included.
- **The lesson:** a model trained only on states the world produces cannot
  tell, by its own dynamics, which states the world cannot produce.
- **In Crafter, H misfires the other way.** It sets aside states with
  creatures and arrows, which move by themselves (cows 2,200–2,700 per
  kind of event), and costs recall on moves.

By the rule, H is dropped and K is the arm judged. Every arm is reported.

| Precision / recall | none (105.4) | H | **K** | HK | Bar |
|---|---|---|---|---|---|
| World 1, lamp goes out (seeds 1021, 7, 3) | 0.66–0.67 / 0.40 | same | **0.67 / 0.40** | same | ≥ 0.9 |
| World 1, lamp lights | 1.0 / 1.0, 1.0 / 1.0, 0.99 / 0.75 | same | 1.0 / 1.0, 1.0 / **0.75**, 1.0 / 0.75 | as K | recall no lower |
| World 1, push, move, blocked | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | |
| Crafter, makes (seeds 1021, 7) | 0.77 / 1.0, 0.84 / 1.0 | 0.80, 0.86 / 1.0 | **0.99 / 1.0, 0.99 / 1.0** | 1.0 / 1.0 | ≥ 0.9 / 0.95 |
| Crafter, blocked moves | 0.95 / 0.96, 0.97 / 0.98 | 0.95–1.0 / 0.94–0.98 | **1.0 / 0.95, 1.0 / 0.95** | 1.0 / 0.94 | ≥ 0.9 |
| Crafter, moves that succeeded | 0.99 / 1.0, 0.92 / 1.0 | 1.0, 0.92 / 0.98–0.99 | **1.0 / 1.0, 1.0 / 1.0** | 1.0 / 0.98–0.99 | ≥ 0.9 |

**Planning** (105.5's planner in world 1, on statements read in sleep):
- 131 of 155 solvable worlds with K, and 131 without it (65.5% of all 200
  test worlds both ways).
- 30 statements either way.

1. **World 1: fail.**
   - **Lamp going out:** 0.67 under every arm. Its false conditions come
     from impossible states, and two copies of the model agree there:
     both learned the same thing and never saw such a state.
   - **K costs recall where a copy is weak:** seed 7, checked against seed
     3, drops to 0.75 on lamps lighting. Seed 3 is the seed that learned
     lamps lighting at only some columns (105.6).
2. **Crafter: pass with K.**
   - Makes go from 0.77 and 0.84 to 0.99, with recall 1.0.
   - Moves reach precision 1.0 and recall 0.95–1.0.
   - K set aside mostly tables (212–213), plants (110–171), water,
     zombies (73) and furnaces (65) placed where the model had no
     experience of them.
3. **Planning: pass** (84.5% of solvable worlds, as before).

## 8. Decision

**Revise** (2026-10-13), by the rule: one world passes.
- **K is the filter to keep.** Copies that agree remove the model's guesses
  where it lacks experience. That fixed Crafter's make conditions, at a
  small cost in recall.
- **H is dropped.** A model's dynamics cannot reveal states it has never
  seen.
- **World 1's impossible states need a test the copies do not share.** The
  copies were trained on the same kind of data and share the same blind
  spot.
- **A candidate for the next card is familiarity by recall.** A replaced
  state counts only if memory holds states like it around the event's
  token: the user's "what memory has shown in that kind of place". A lit
  lamp with no box beside it is in no memory. Recall is part of the
  brain, so it carries to every world.
