---
id: "140"
title: which period of the past is the present like — stored periods weighted by how well each has just predicted
rung: 3
serves: [P4, P12]
status: done
verdict: fail
arch_version: og4
date: 2026-11-13
---

# 140: which period of the past is the present like?

Roadmap step 3, rung 3. After
[card 139](../139-an-answer-that-knows-when-not-to-trust-itself/card.md)
the largest gap left on the memory world is knowledge that has gone
out of date: the child whose parent is swapped for another switch
every 300 episodes (62% of the gap closed on warm touches; the other
factors 75 to 100%). The user (2026-11-13) approved it as the next
target.

In the scored episodes the world is wired first as it was 300
episodes earlier, not as it was most recently, so "recent over old"
would be wrong for the first stretch. What is needed is to tell
which part of the past the present resembles.

Biological counterpart: context inference. Animals treat a change in
what follows what as a new context and bring back an old one when
its regularities return (latent cause inference, Gershman, Blei and
Niv 2010; hippocampal remapping).

## 1. Question

If the store is also read period by period, and each period is
trusted by how well it has just predicted what the agent saw, does
memory follow a change of the world's rules, including a return to
old rules, without harm where nothing changes? P4.

## 2. What changes

One component: which stored touches answer. Records, key and the
25-nearest share are card 138's kept arm.

```
138:      answer = share of the 25 nearest stored touches of the object, whole store
140:      experts = "all" (as 138) + one per stored period of 80 episodes (25 nearest
                    within the period; needs 25 touches of the object)
          weight  = prior x exp(-sum of the expert's log losses on the object's last 20
                    touches, whose outcomes the agent has seen)
                    prior: half on "all", half spread over the periods
          answer  = weighted mean of the experts' probabilities
```

This is Bayesian averaging over "the present is like period j",
with a short memory for the evidence. Nothing is trained and nothing
says which object is rewired; the rule runs for every object.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| Outcomes are seen soon after the answer | yes | yes | yes | yes | general |
| Rules hold for stretches, then change | seasons, updates | per map | per level | tools and places change | every 300 episodes, one object |
| A period of 80 episodes is a usable grain | a declared number | the same | the same | the same | declared; not aligned with the changes (300) |

## 3. Dependencies

Cards 136.1, 137, 138.

## 4. Data check

The rewired child has about 600 warm scored touches a world, about
80 of them in the first 40 episodes after a rewiring (episodes 600
and 900).

## 5. Feasibility gate

- **Upper bound:** the observer. **Trivial baseline:** card 138's
  answer.
- **Reference that knows which past is like now:** for the rewired
  child the store is cut down, using the world's truth, to the
  episodes wired as now. Measured in the same run as the arms. The
  gate passes if the reference is at least 10 points of gap closed
  above card 138's answer on warm touches of the rewired child; if
  it is not, the card is void.

## 6. Success criteria and prediction

Three worlds, mean. Bars are set against the reference, before it
is measured.

1. **It follows the change:** rewired, warm: at least half of the
   way from card 138's answer to the reference, in gap closed.
2. **No harm where nothing changes:** every other warm factor at
   most 1 point under card 138's answer; cold carry at least 99%.
3. **Soon after a change:** in the first 40 episodes after a
   rewiring, rewired, warm: at least a third of the way from card
   138's answer to the reference, in log loss.

Prediction: 1 passes; 2 is the risk (thin periods answering for
objects that never change); 3 uncertain.

## 7. Result

Run 2026-11-13, `tools/card140/periods.py world=0|1|2` (75 seconds
each), `runs/140/periods.json`, numbers in
[results.json](results.json).

**The gate did not pass, by a hair:** on warm touches of the
rewired child the reference closes 72.3% of the gap and card 138's
answer 62.4%, 9.9 points where 10 were asked. By its own rule the
card is void. The arm is reported all the same, and it fails every
bar.

| Warm touches, gap closed | Card 138 | Periods (this card) | Reference |
|---|---|---|---|
| rewired | 62.4% | 63.6% | 72.3% |
| rewired, first 40 episodes after a change (log loss) | 0.387 | 0.396 | 0.453 |
| drift slow | 90.5% | 84.8% | |
| drift fast | 82.4% | 76.3% | |
| chance | 75.2% | 69.1% | |
| two | 79.7% | 76.0% | |
| timer | 93.1% | 90.5% | |
| random | 99.9% | 98.3% | |

1. **It follows the change: fail.** 63.6%, where half of the way to
   the reference is 67.4%.
2. **No harm: fail.** Six warm factors lose 2 to 6 points (allowed
   1). The "all" expert holds only 57 to 70% of the weight on the
   children whose outcomes are partly chance: over 20 touches a
   thin period is right by luck often enough to be believed.
3. **Soon after a change: not defined.** The reference is worse
   than the baseline there (0.453 against 0.387).

**Where the loss sits** (a check afterwards, not a declared arm;
`tools/card140/why.py`; rewired, warm, mean of three worlds):

| | Whole store answers | Only episodes wired as now answer |
|---|---|---|
| Key chosen and weighted on the whole store | 62% | 72% |
| Key = own record + the true parent now | 73% | 85% |

The reference was badly built: it kept a key chosen on both wirings
together. Which stored episodes answer and which records are in the
key each cost about 10 points, and both together 23. Mixing the
answers of periods cannot recover the second half; each period
would need its own choice of records.

## 8. Decision

**Stop.** Mixing the answers of stored periods is dropped: it does
not follow the change and it harms what does not change. What the
check shows is kept for a next card: a change of rules changes the
condition itself (which object the outcome depends on), so the
period that resembles the present has to supply its own key, not
only its own cases; and evidence over 20 touches is too little to
choose among many periods when outcomes are partly chance.
