---
id: "110"
title: confidence — does System 1 know where its predictions apply?
rung: 1
serves: [P5, P12, P4]
status: done
verdict: fail
arch_version: 27
date: 2026-10-17
---

# 110: confidence

Roadmap step 2 (conditions of one-step effects). Proposed at the user's
request, 2026-10-17: "propose a card to start working on confidence".
Crafter and world 1 are development worlds here. The Craftium follow-up is
110.C, run once Craftium's adapter gives tokens.

## 1. Question

System 1 gives a probability for each token's next code (105.8). Can each
prediction also carry a confidence that is calibrated on real states and
falls where the model has no experience? And does reading conditions only
where that confidence is high fix the false conditions that two agreeing
copies could not (106)? Rung 1. Serves P5 (calibrated confidence, and a sense
of when a prediction applies), then P12 and P4 through the conditions read.

**Why this, and why now.** Every part above System 1 multiplies its
probabilities:
- conditions (106's filter);
- belief (109, scored in nats);
- the rule for recall against networks;
- the planner's evaluation;
- version 27's exploration.

A probability that is confident where the model has never been breaks all of
them, whatever architecture sits above. Card 106 showed the gap:
- In world 1, 493 of 501 false conditions of a lamp going out come from
  impossible states, such as a lit lamp with no box beside it.
- Two copies of the model agree on those states, so precision stays at
  0.67.

## 2. What changes

One component: the confidence used to decide which replaced states count
when conditions are read. It replaces 106's filter (K). The transition
model and the interventions are unchanged (105.4, 106).

```
106 (K):  a replaced state counts if two trained copies predict the same next code
110 (R):  confidence = p(next code) × a(state)
          p: the model's own probability (105.8), temperature-scaled on a held-out split of real states
          a: familiarity by recall: the distance from the queried token's vector in the model's last
             layer to its k nearest stored vectors (from training, the same action), turned into the
             probability that the prediction is right by a two-parameter logistic fitted on held-out
             real states; k and the kernel width are chosen by leave-one-out likelihood (Duin 1976)
          a replaced state counts if its confidence is above a threshold that keeps 95% of held-out
          real states (fixed before scoring)
arms:     P (p alone), K (106), R, RK (R and K both)
```

**Why recall on the model's own vector.** It is the user's "what memory has
shown in that kind of place" (106's decision). It reads the one latent
space (C2), needs no second network, and keeps every experience (recall
replaces counting). Copies trained on the same data share blind spots, but
a state none of them has seen is far from every stored one.

**The known risk.** A network can map a new state onto a familiar vector:
recall over a network's context cannot see what the network has not learned
(LESSONS, 102.1). If the lit lamp with no box sits among stored lit lamps,
R fails, and the next card compares the input tokens instead.

### Portability check (CHARTER rule 9)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The model's per-token vectors can be stored and searched | yes | yes | yes | yes, with a bounded memory | general (brain) |
| Familiarity is judged per action | yes | yes | yes | continuous actions need a kernel over actions | declared; 110.C keeps discrete actions |
| Held-out real states exist to fit the scale and the threshold | yes | yes | yes | yes | general |
| States that are impossible or out of experience are labelled by the world's rule | scoring only | scoring only | scoring only | scoring only | evaluator, never the learner |

## 3. Dependencies

- **Models:**
  - world 1: 105.6's, seeds 1021, 7 and 3;
  - Crafter: 105.10's, seeds 1021 and 7.

  None is retrained.
- **Interventions, truth and events:** `tools/card105.4/`, `tools/card106/`.
- **Literature** (added to LITERATURE when approved):
  - Guo et al. 2017: temperature scaling and expected calibration error;
  - Sun et al. 2022: k-nearest-neighbour distance in a network's features
    detects inputs out of distribution;
  - Lakshminarayanan et al. 2017 (already listed): K;
  - Duin 1976 (already listed): the width chosen by leave-one-out.

## 4. Data check

- **World 1:** 105.4's held-out events (213 lamps lighting, 141 going out,
  300 each of pushes, moves and blocked moves). There are 39,744 replaced
  states the world's rule calls impossible and 206,835 possible ones (106).
- **Crafter:** 232 makes, 300 blocked moves and 300 moves that succeeded,
  plus held-out real steps of every action for criterion 1.
- **Memory size:** 132,785 training transitions times the tokens in each
  state. Before scoring, the run counts the vectors and checks that they fit
  in 4 GB at 16 bits. If they do not, the memory is subsampled within each
  (action, next code), keeping every token whose code changed (rule 5: no
  square-size arrays; the search runs in chunks on the GPU).

## 5. Feasibility gate

- **Upper bound:** an oracle familiarity is 1 if the queried token's code
  and every code in its 5 × 5 surroundings occur together in training, and
  0 otherwise (scoring only).
  - Its precision on world 1's lamp going out shows whether criterion 3
    can be met.
  - Its AUROC for errors sets criterion 2's bar.
- **Trivial baselines:**
  - P: expected to be confident on the impossible states;
  - K: 106's numbers (lamp going out 0.67).

The gate passes if the oracle reaches precision ≥ 0.9 on a lamp going out,
and P and K do not.

Result of the gate, before the main run:

## 6. Success criteria and prediction

All three must hold for one arm that contains R, in every seed of both
worlds.

1. **Calibrated on real states.**
   - **Measure:** the expected calibration error (15 bins) of the predicted
     next code on held-out real steps.
   - **Bar:** at most 0.05 for each kind of event with at least 100
     occurrences (move, blocked move, make, do, no-op, a creature's step).
   - **Why per kind:** an overall score hides rare changes, which is where
     overconfidence would be.
2. **Confidence falls where the prediction is wrong.**
   - **Measure:** the AUROC of the confidence for "the predicted code is
     wrong", against the world's rule. It is scored on the replaced states
     105.4 asks (world 1: every kind of event; Crafter: makes and moves).
   - **Bar:** at least 0.9, or the oracle's AUROC minus 0.05 if the oracle
     is below 0.95 (fixed at the gate). It must also be at least 0.05 above
     P and K.
3. **It changes the conditions read (C4).**
   - Conditions counted through the confidence reach precision ≥ 0.9 for
     every kind of event, a lamp going out included.
   - Recall stays within 0.05 of 106's K: world 1's moves and pushes 1.0;
     Crafter's makes 1.0 and moves at least 0.95.

**Reported:**
- reliability diagrams;
- the confidence of real, possible-but-unseen and impossible states;
- criterion 3 against the memory's number of distinct occurrences of each
  event (10, 30, 100, 300; P19), by subsampling the memory without
  retraining;
- the cost per query.

**Prediction:**
- **P** is calibrated on real states, since 105.8's loss is a proper
  scoring rule, but confident on impossible ones.
- **R** separates world 1's impossible states (AUROC about 0.95), and the
  lamp going out reaches about 0.95 precision.
- **Crafter is harder** (AUROC about 0.85), because many replacements land
  near familiar vectors; **RK** passes there.
- **Most likely miss:** the lamp's vector without its box sits among stored
  lit lamps (the risk in section 2).

**Decision rules:**
- **Keep** (R, or RK, replaces K as step 2's filter, and its confidence
  becomes System 1's output for every consumer) if 1–3 hold.
- **Revise** if one world passes, or if 2 passes and 3 fails.
- **Stop** if R's AUROC on world 1's impossible states is not at least 0.1
  above K's. Familiarity is then not in the model's vector, and the next
  card compares input tokens.

**Budget:** no training. Extracting vectors, the search, the fits and the
scoring take about 3 minutes per seed, 5 seeds, so about 15 minutes in
short capped runs (CHARTER rule 5), after a one-seed smoke test.

## 7. Result

Run 2026-10-17 ([results.json](results.json); `tools/card110/world1.py`,
`crafter.py`, `common.py`). Each run took 1–2 minutes, capped.

**Smoke-test findings, kept as written:**
- **World 1's model makes 1–4 errors in 63,000 held-out real tiles.** So
  section 2's logistic, which turns distance into "probability right", has
  almost nothing to fit there.
- **In Crafter, distance does not predict real errors** (AUROC 0.52–0.53).
  The fit came out as a steep step.
- The mechanism was not changed after the smoke test. The raw distance is
  reported beside it, since it does not depend on the fit.

**Events:** in world 1 the script drew from the random generator before
drawing events, so they are a different draw of 105.4's events than 106's,
not the same one. The arm with no filter matches 106's numbers (lamp going
out 0.66 / 0.40). Crafter's events are 106's.

**Memory:** 2.8 million vectors in world 1 and 3.4 million in Crafter (all
tokens of 40,000 states, and every token whose code changed).

**Gate: pass.**
- The oracle reaches precision 1.0 on a lamp going out; P and K reach 0.67.
- But the oracle keeps very few true conditions (recall 0.03–0.48): true
  conditions are found by replacements that make new surroundings.
- The oracle's AUROC is 0.62 (world 1) and 0.53 (Crafter), so criterion
  2's bar is 0.9, with the 0.05 margin over P and K.

| | World 1 (seeds 1021, 7, 3) | Crafter (seeds 1021, 7) | Bar |
|---|---|---|---|
| 1. Worst calibration error, P / R | 0.002–0.005 / 0.001–0.002 | changed tokens 0.13–0.45, creatures 0.03–0.11 (both arms) | ≤ 0.05 per kind |
| 2. AUROC for a wrong prediction: P, K, **R**, RK | 0.83–0.85, 0.58–0.61, **0.79–0.84**, 0.83 | 0.89–0.95, 0.88–0.94, **0.96–0.98**, 0.98–0.99 | ≥ 0.9 and 0.05 above P and K |
| 2'. Raw distance, a lamp going out | AUROC 0.23–0.34 (wrong predictions sit *nearer* memory) | | diagnostic |
| 2''. R on impossible states, against K | 0.53–0.55 against 0.50 | | stop rule: ≥ K + 0.1 |
| 3. Lamp going out, R (precision / recall) | 0.63–0.67 / 0.33–0.40 | | ≥ 0.9 |
| 3. Makes, R | | 0.99 / 0.99 (K: 0.99 / 1.0) | ≥ 0.9, recall within 0.05 of K |
| 3. Moves blocked, R | R recall 0.44–1.0 | 1.0 / 0.95–0.96 | |
| 3. Moves succeeded, R | R recall 0.07–0.55 | 1.0 / **0.64–0.83** (K: 1.0 / 1.0) | |

1. **Calibrated on real states: pass in world 1, fail in Crafter.**
   - **Tokens whose code changed** are badly calibrated (0.13–0.45), but
     this split is defined by the outcome. A predictor that is calibrated
     on random changes fails it too (sapling growth, for example, happens
     with probability 0.1 per try). The test was wrong for this part.
   - **Creatures**, a split defined by the input, also exceed 0.05 for
     do, noop and seed 1021's moves (0.06–0.11).
   - R changes nothing here: on real states its familiarity term is about
     1.
2. **Confidence falls where the prediction is wrong: fail.**
   - **World 1:** R is below P (0.79–0.84 against 0.83–0.85).
   - **Crafter:** R is above the bar (0.96–0.98), but only seed 7 clears
     P by 0.05 (+0.07; seed 1021 +0.03). With K added, RK reaches
     0.98–0.99 (+0.047 and +0.098 over P).
3. **It changes the conditions read: fail.**
   - **World 1:** a lamp going out is unchanged (0.63–0.67), and R cuts
     recall on moves and pushes.
   - **Crafter:** makes and blocked moves match K, but moves that
     succeeded lose recall (0.64–0.83 against 1.0).

**P19** (R's conditions against the memory's number of occurrences of the
kind, 10 to 300): flat for makes and blocked moves. For moves that
succeeded it is erratic (recall 0.0–1.0 in world 1, 0.20–0.79 in Crafter),
because a smaller memory moves every distance and the threshold is a
percentile.

**Why world 1 fails.** The model's vector for a lit lamp with no box
beside it lands among stored lit lamps that stay lit. Those are the
commonest lit lamps in memory, so the wrong prediction looks *more*
familiar than right ones (AUROC 0.23–0.34). The model never learned that
the box matters for staying lit. So its own vector cannot show the box's
absence: LESSONS' "recall that compares by a network's context cannot see
a condition the network has not learned", now in a second world.

**Why Crafter half-works.** Its false conditions came from tokens placed
where the model has no experience (106: tables, zombies and plants 2–4
tiles away). There the vector does move away from memory. The raw
distance alone gives AUROC 0.91–0.95, above the two copies (0.75–0.86 on
moves). But a percentile threshold on familiarity also removes the
unfamiliar replacements that reveal true conditions (a tree or water put
on the target of a move).

## 8. Decision

**Stop** (2026-10-17), by the rule: R's AUROC on world 1's impossible
states is 0.03–0.05 above K's, not 0.1.
- **Familiarity in the model's own vector space cannot replace the copies
  as the filter.** It fails exactly where the model has not learned the
  condition. That is the case a confidence most needs to catch.
- **It is informative where the model's input is new** (Crafter, AUROC
  0.96–0.98, with K 0.98–0.99). So it is worth keeping as a signal, not as
  a filter.
- **Two findings reach beyond this card:**
  - **Applicability cannot be learned from the model's errors on real
    states.** Those errors are unrelated to distance (AUROC 0.53), and
    world 1 has almost none.
  - **Setting aside unfamiliar states costs true conditions,** even with
    an oracle (recall 0.03–0.48). So a low confidence should send the
    agent to try the state (P15; "trying confirms or corrects what recall
    infers", CHARTER), not drop the condition.
- **The card's own next step,** familiarity over input tokens, is
  answered in part by the oracle: exact surroundings keep precision but
  lose most recall.
- **Criterion 1's split by changed tokens conditions on the outcome.** A
  later calibration test splits only by what the input shows.
