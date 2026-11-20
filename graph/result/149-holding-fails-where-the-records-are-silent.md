---
id: result/149-holding-fails-where-the-records-are-silent
type: result
title: the energy answers arbitrarily about a combination its records do not hold, and settling proposes one
card: '149'
model: model/og4
env: environment/sprite-world
measure: percent of 200 test tasks (hold object X) reached within 60 steps, 3,000 steps of experience, mean of 5 streams
value: 58.8
against: 93.0
n: 200
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/propose-then-verify-with-one-energy
- rel: tested
  to: principle/action-as-output-token
---
[Card 149](../../experiments/149-a-sequence-of-actions-by-recurrence/card.md), the first task of criterion 1: 99.0, 0.0, 0.0, 98.0 and 97.0% on the five streams. For "hold X" while touching it the energy proposes closing the grip without moving; the records hold that 0 or 1 times in about 200 takes; the forward prediction for it is "held" in three streams and "not held" in two, and the task succeeds or fails accordingly. An earlier design, which read whether an effect can happen from the energy's probability for the wanted token, failed the same way (0 to 98% on validation).
