---
id: result/148-whole-records-need-ten-times-the-experience
type: result
title: whole records in a Hopfield memory need about ten times the experience of pairwise couplings
card: '148'
model: model/og4
env: environment/sprite-world
measure: percent of test steps in which the completed action gives exactly the wanted effect, stored records, 100 steps of experience, mean of 5 streams
value: 77.5
against: 80.0
n: 6000
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/action-as-output-token
- rel: tested
  to: principle/store-everything-read-by-similarity
---
Criterion 2 of [card 148](../../experiments/energy-modules/148-the-action-as-an-output-token/card.md) for the records arm: every (step, object) stored whole and retrieved by the softmax of matching tokens gives 77.5% after 100 steps (couplings 95.6%), 93.6% after 300 and 99.3% after 1,000. A record is one whole combination of move and grip; the couplings hold "move by what the hand should move by" and "close if something should be held" as separate facts (cross-axis couplings are zero) and produce combinations never seen.
