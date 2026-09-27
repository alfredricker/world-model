---
id: "010"
title: condition logic
rung: 0
serves: [P12, P19, P4, P6]
status: draft   # draft | approved | gated | running | done | abandoned
verdict:
arch_version: 0
date: 2026-09-27
---

# 010: condition logic

## 1. Question

Does our definition of a condition pick out the right conditions in worlds
where plausible definitions disagree (a condition that can be undone, two
alternative routes, two prerequisites needed together, an irrelevant
lasting change), with no hand-picked thresholds, from few examples; and can
a network learn each of those logical requirements from pixels? The
key-door room cannot tell definitions apart: its key is necessary,
permanent and never optional. Card 009 (undoability) was shelved for
exactly that reason. Serves P12 (conditions), P19 (few examples), P4
(ignore what does not matter) and P6 (predict what an action will do).

## 2. The definition under test

Agreed with the user 2026-09-27 (discussion in the session; see card 003
section 2 for the theory it refines).

- **Relative to an attempt.** Conditions belong to a particular attempted
  action (later: a skill) and its intended outcome. "Toggle now" needs the
  key; "do whatever it takes" does not.
- **Structure.** Whether the attempt succeeds is an **or** of rules, each
  an **and** of conditions, each condition a yes/no detector or its
  negation: a decision list, each rule with its own success probability and
  a leak probability when no rule applies. Any true/false requirement can
  be written this way.
- **Admission by evidence, no thresholds.** A condition or rule is added
  only if it raises the Bayesian evidence of the observed attempts: the
  probability of the successes and failures with each rule's success rate
  averaged out (Beta(1,1) priors), minus the cost of describing the added
  structure (log of the number of candidate conditions per condition, a
  fixed cost per rule). This is a Bayesian rule list (Letham et al. 2015)
  and the minimum description length principle. It replaces the finder's
  four hand-set cut-offs (success rate ≥ 0.9, removal ≥ 0.1, ≥ 5% of
  successes covered, keep ≥ 95%). What remains chosen by hand is the
  prior; it is stated as a probability and its weight falls as data grow.
- **Search.** Greedy, one condition at a time, with a two-condition
  look-ahead so that conditions useful only together are not missed.
- **Reported, not used to decide:** each condition's removal effect with
  its rule's other conditions held (card 003), and its evidence in bits.

## 3. What changes

**Worlds.** Card 003's room, extended: six actions (left, right, forward,
pickup, **drop**, toggle); a **switch** in the left room that toggles on
and off (drawn as a ball, grey off, yellow on); a **vase** in the left room
that a toggle breaks for good (drawn as a box; gone once broken), which no
goal depends on. Doors check their requirement at every opening (closing
and reopening needs it again), so each world has one clear rule set. Five
worlds, all with drop and the vase:

| World | Door opens on toggle, facing it, when | Expected rules for "door open" |
|---|---|---|
| W0 key | holding the matching key | toggle ∧ facing door ∧ matching key |
| W1 switch | the switch is on (undoable) | toggle ∧ facing door ∧ switch on |
| W2 either | matching key **or** switch on | two rules: …key; …switch on |
| W3 both | matching key **and** switch on | toggle ∧ facing door ∧ matching key ∧ switch on |
| W4 no drop | as card 003 (no drop action, no switch) | card 003's rule |

Other goals analysed in every world: holding the matching key (pickup ∧
facing the matching key ∧ empty hands), switch on (toggle ∧ facing the
switch), on the goal square (forward ∧ facing it). The vase must appear in
no rule. The candidate vocabulary is card 003's variables plus switch
state, facing the switch, vase broken, facing the vase.

**Part A: exact.** The evidence finder on random play with the simulator's
variables, as in card 003.

**Part B: learned from pixels.** Per world, card 005's network (speed
settings of card 008) learns from frames whether each action achieves
each goal (supplied success signals, C1). The evidence finder then runs on
the network's predictions (A > 0.5) on new layouts, with the simulator's
variables as vocabulary, as card 005 did. Finding the detectors themselves
without that vocabulary is card 011, on these same worlds.

## 4. Dependencies

Card 003 (world, exact analysis), card 005 (frame learning), card 008
(speed settings). Bayesian rule lists: Letham et al. 2015 (to add to
LITERATURE). Code: `envs/keydoor.py` extended (a test checks the symbolic
step against a MiniGrid version and the renderer against MiniGrid tiles).

## 5. Data check and feasibility gate

Random play, 10k episodes per world. Each expected rule must fire (door
opened through it) at least 300 times in training (card 005 needed about
300 unlocks); W3 needs key and switch together and is the rarest. If a
world falls short, add play starts (some episodes begin holding a key or
with the switch on) and report both.

- **Upper bound:** the simulator's own rules.
- **Trivial baseline:** card 005's threshold finder on the same data, and
  the frequency ranking of card 003 (which variables are common just before
  success).

## 6. Success criteria and prediction

1. **Exact, all worlds (Part A):** the evidence finder gives exactly the
   expected rules for every analysed goal in W0–W4, with the vase in no
   rule, using no thresholds.
2. **Few examples (Part A):** with training cut to 5, 10, 30 and 100
   successes of the rarest rule, exact recovery in ≥ 90% of 20 subsamples
   at 30 (curve reported).
3. **Learnable from pixels (Part B):** on new layouts of every world, the
   rules read from the network's predictions equal those from the true
   outcomes, including W2's two rules and W3's four-part rule.

Prediction: 1 passes; the look-ahead is not needed here because each
"and" condition alone already raises the success rate. 2: recovery by 10
successes for single rules, 30 for W3. 3: passes in W0, W1, W4; W3 is
the risk (the network must see key and switch together, the rarest case).
Runtime: Part A minutes; Part B about 8 minutes per world, about 40 in
all, over 30 minutes, so it needs the user's approval to run here.

## 7. Result

## 8. Decision
