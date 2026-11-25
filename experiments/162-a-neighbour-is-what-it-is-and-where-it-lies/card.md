---
id: "162"
title: a neighbour is what it is and where it lies — one effect network in which a move learned in one heading carries to the others
rung: 5
serves: [P3, P6, P12]
status: done
verdict: pass
arch_version: og4
date: 2026-11-24
---

# 162: a neighbour is what it is and where it lies

MiniGrid, the agent of
[card 161](../161-conditions-asked-of-the-effect-model/card.md): it learns
what actions do from reward-free random play and reaches a goal by
chaining conditions backward from it.

Card 161 left KeyCorridor at 66.7% and traced it to the learned effect of
the agent's own move. That effect has separate weights per heading, and
each neighbouring cell enters as one input per (position, code) pair. So
"an open purple door can be walked through" is learned once per heading
and per colour. Random play never walked west through open doors of some
colours, and the agent then could not imagine doing so.

The user's rule (2026-11-24): build from how parts are similar. Two
things are the same here whatever the heading: the role, "the cell
ahead", and what fills it, "a code that can be walked through".

Biological counterpart: the brain codes what an object is and where it
is in separate pathways and binds them by attention (Ungerleider and
Mishkin 1982; Treisman and Gelade 1980); the entorhinal cortex keeps
object and place codes apart and the hippocampus joins them
(Tolman-Eichenbaum Machine, LITERATURE).

## 1. Question

If every token in a situation enters the effect network as two parts,
what it is and where it lies relative to the token asked about, does a
move learned in some headings carry to the others, and does the agent
then solve KeyCorridor? Rung 5. P3 (roles separate from what fills
them), P6, P12.

## 2. What changes

One component: the effect network's inputs and how it combines them.

```
161:  the agent's move:  own weights per heading; one input per (position, code) of the 8 cells around it
      a cell's change:   one input for the agent's place and heading together; one per token out of view
162:  every token in the situation = what (an embedding of its code, one table for every use)
                                   + where (an embedding of its offset from the token asked about, or "far",
                                            or "out of view by action a")
      attention: where a token lies, with the token asked about and the action, decides how much it counts;
                 what it is decides what it contributes.  Then a small network gives the next state.
```

A token contributes what it is and, apart from that, the bare fact
that something lies there (card 160 gave a token out of view twice the
same way). No rotation is built in. Which offset matters for which
heading is learned from the cases random play covers well (empty cells
and walls in all four headings), and must carry to the rare ones.

**The card tests the new network for the agent's move** (arm A); a cell's
change stays with card 160's perceptron, as in card 161. One network for
the agent and every cell alike (arm B) was built too and **failed the
gate** (section 5); it is not tested here.

Declared (C3): attention's keys come from where a token lies only; a
relation is an offset on the grid within two cells, "far", or "out of
view by action a".

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Tokens with an identity are supplied | needs a front end (C1 allows it) | the same | holds (cells) | needs a front end | declared, as in card 160 |
| Where a token lies is one of a few offsets on a grid | no: a continuous offset in three dimensions | holds | holds | no: continuous | declared; the embedding of an offset index must become a function of the offset vector |
| The agent's heading is one of four | no: continuous | four | none | no | declared, with the row above |
| The whole grid is in view | no | no | holds | no | declared, as in card 160 |

## 3. Dependencies

- Cards 160 and 161: tokens, memory, three random-play experiences,
  the chaining agent, the perceptron for cells (arm A), the rules agent.
- Attention with keys and values (Vaswani et al. 2017), an established
  method; binding what to where (Tolman-Eichenbaum Machine, LITERATURE).

## 4. Data check

Card 161's diagnostics, on layouts outside every test. Random play holds
no forward move west into an open door of 2 to 4 colours of 6, by
experience. KeyCorridor needs a door passed heading west in most layouts.
With MiniGrid's rule for the agent's move only, the agent solves 40 and
38 of 40 (learned move: 36 and 28).

## 5. Feasibility gate

- **Upper bound:** MiniGrid's rule for the agent's move with everything
  else learned: 40 and 38 of 40 on KeyCorridor (section 4).
- **Trivial baseline:** card 161's agent.

Result (2026-11-24; 40 layouts per task outside every test, seeds
2000–2039; the networks fitted here are the ones tested), by experience
1, 2, 3:

| | Card 161's agent | Arm A |
|---|---|---|
| KeyCorridor solved, of 40 | 36, 25, 28 | 40, 34, 38 |
| A door imagined open can be walked through heading west, of 149 approaches | 106, 69, 53 | 149, 149, 149 |
| "Does forward move the agent", (code, heading) pairs never tried | 4 of 23, 4 of 26, 3 of 26 | 23 of 23, 26 of 26, 26 of 26 |
| The same, pairs tried | all right | all right |
| UnlockPickup, BlockedUnlockPickup, of 40 | (card 161's gate, experience 1: 30, 33) | 30, 33; 34, 35; 30, 32 |
| DoorKey-8x8, of 40 | | 40, 40, 40 |

Passed for arm A, in every experience. Card 161's move effect even says
the agent walks through a locked door in headings it never tried.

**Arm B failed the gate.** Its first fit had not converged; fitted on
every row at every step it fits the experience better than the perceptron
(21 nats against about 105), and still carries to untried cases worse on
both sides: "will the pickup happen" 87% right in experience 3 (the
perceptron 97%), moves never tried 20 to 22 of 26 (arm A 26 of 26),
UnlockPickup 13 and then 28 of 40 after a repair (arm A 30). I do not
know why learning both together hurts what each learns alone. It needs a
card of its own. The trial fits are in `runs/162/first_fit/` and
`second_fit/`.

## 6. Success criteria and prediction

Test: card 160's five tasks, 100 fresh layouts each (seeds 6000–6099,
never run), three experiences, code and fitted networks frozen before the
run. Comparison: card 161's agent on the same layouts.

1. **Success:** mean over the five tasks at least 90% in each experience
   (card 161: 87.6, 86.6, 86.2), and, on the mean of the three
   experiences, no task more than 5 points under card 161's agent.
2. **KeyCorridor:** at least 90% on the mean of three experiences and at
   least 85% in each (card 161: 66.7; 83, 61, 56).
3. **The move carries to what was never tried:** with each known code
   ahead and each heading, "does forward move the agent" is right, by
   MiniGrid's rules, for at least 95% of the (code, heading) pairs the
   experience never holds, in each experience.

Criterion 3's measure does not depend on the layouts; its value for the
frozen networks is already in the gate's table (all pairs right).

Prediction: 1 and 3 pass. 2 is near its bar: experience 2 solved 34 of 40
in the gate, exactly 85%. The pickup tasks stay near 84%, since nothing
in this card touches them.

Budget: no further fits. Six test runs (two agents, three experiences),
each split so that no run exceeds 10 minutes; about 20 minutes in all,
three at a time.

## 7. Result

Run 2026-11-24, approved by the user. Code and networks frozen before the
test (`runs/162/frozen.txt`, unchanged after it); numbers in
[results.json](results.json); `tools/card162/` (`net.py`, `ev.py`).

Percent of 100 fresh layouts solved, mean of three experiences (each
experience in brackets):

| | DoorKey-8x8 | DoorKey-16x16 | UnlockPickup | BlockedUnlockPickup | KeyCorridorS3R3 | Mean |
|---|---|---|---|---|---|---|
| Card 161's agent | 100 | 100 | 83.7 | 85.0 | 63.0 (78, 58, 53) | 86.3 (87.2, 86.0, 85.8) |
| **Arm A** | 100 | 100 | 83.7 | 85.0 | **94.0** (96, 93, 93) | **92.5** (90.8, 93.0, 93.8) |
| Told MiniGrid's rules (card 160) | 100 | 100 | 100 | 100 | 100 | 100 |

| Criterion | Result | Verdict |
|---|---|---|
| 1. Mean at least 90% in each experience; no task more than 5 points under card 161's agent | 90.8, 93.0, 93.8; four tasks identical, KeyCorridor 31 points higher | **pass** |
| 2. KeyCorridor at least 90% on the mean and 85% in each experience | 94.0; 96, 93, 93 | **pass** |
| 3. "Does forward move the agent" right for at least 95% of pairs never tried | 23 of 23, 26 of 26, 26 of 26 (card 161's move effect: 4, 4 and 3) | **pass** |

**Verdict: pass, 3 of 3.** Criterion 2 passed with more room than I
predicted.

- **A move learned in some headings carries to the others.** With what a
  neighbour is kept apart from where it lies, the network is right about
  every (code, heading) pair it never tried, with no rotation built in.
  The old move effect was right about one in seven.
- **KeyCorridor rises from 63.0% to 94.0%,** and the agent solves it in 34
  steps where it took 57. Layout by layout, arm A solves 99 that card
  161's agent fails and fails 6 that it solves.
- **The other four tasks are identical to the layout,** as they should be:
  nothing they use changed.
- **One network for every token is not shown.** Arm B failed the gate
  (section 5) and was not tested.

Open: in KeyCorridor 47% of steps still have no plan. On the two pickup
tasks 93 of 94 failures end with no plan, a random action and the box
destroyed, as in card 161. Limits: tokens, the full view and the chaining
procedure are supplied; where a token lies is an offset on a grid.

## 8. Decision

**Keep.** The agent's move is predicted by the network in which a token
is what it is and where it lies. Mean success is 92.5% on fresh layouts
against 86.3%, and against 70.3% two cards ago. Next, with the user's
go-ahead: the agent holds its expected next event and notices when it does
not come (the user, 2026-11-24), which is where "no plan, then a random
action" is replaced; and one network for every token, as a card of its
own that starts from why learning both together hurt.
