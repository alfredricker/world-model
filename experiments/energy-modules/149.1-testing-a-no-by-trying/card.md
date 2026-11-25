---
id: "149.1"
title: the agent tests a "no" from the couplings by trying it once
rung: 2
serves: [P7]
status: abandoned
verdict:
arch_version: og4
date: 2026-11-20
---

# 149.1: testing a "no" by trying

The revision that [card 149](../149-a-sequence-of-actions-by-recurrence/card.md)
decided; the user agreed to run it ("sure", 2026-11-19). **Stopped by the
user on 2026-11-20 before its test run** ("ok that isn't the right solution
then"), after I reported that its first form had made things worse. The
user's next direction is energy modules in layers, updating in parallel.
This card holds what was seen on validation so that it is not lost. No
criterion was judged.

## 1. Question

Does card 149's agent stop failing where its records are silent, if it
tries for itself what the couplings say will not work?

## 2. What changed, in two forms

1. **Records overrule the couplings (dropped).** Every (object state,
   action) of the experience was stored with what it did, and a stored
   record decided before the couplings did. On the validation stream this
   was worse than card 149: task 1 with 300 steps 33.5% against 98.0%;
   task 2 with 1,000 steps 81.5% against 91.5%. A record keyed on one
   object's state misses context: a grip closing on a touched object does
   nothing when the hand already holds another, and one such record
   vetoed taking.
2. **A "no" is tried once (running when stopped).** No stored records.
   When the couplings say the proposed action will not give the wanted
   effect and the agent has not tried that combination in this episode, it
   tries it; what it sees decides for the rest of the episode. If the "no"
   is about "would it work if the object were touched", it goes and
   touches the object to try.

Portability check: as card 149, plus one assumption: a try costs a step
and nothing lasting (not true of a robot that can drop or break things).

## 7. Result (validation tasks only; streams 200, 201 and part of 202)

Percent of 200 validation tasks reached; card 149's agent → form 2.

| Stream, steps of experience | Task 1, hold X | Task 2, X at G and let go |
|---|---|---|
| 200, 300 | 98.0 → 98.0 | 38.5 → 46.5 |
| 200, 3,000 | 97.5 → 97.5 | 98.0 → 98.5 |
| 201, 300 | 0.0 → 98.5 | 77.5 → 99.0 |
| 201, 3,000 | 0.0 → 99.0 | 94.0 → 98.0 |
| 202, 300 | 61.5 → 92.0 | 55.0 → 56.0 |

Stream 201 is one where card 149's agent never holds the object. Form 2
holds it in 98.5 to 99.0% there, at 1.23 to 1.25 times the solver's steps
(card 149's bar: 1.5). Two of the five validation streams and the whole
test were not run. Code: `tools/card149.1/sequence.py`; logs `runs/149.1/`.

## 8. Decision

**Stop** (the user). Form 1 is not to be revived. Form 2 was not judged.
