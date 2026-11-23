---
id: result/159-learned-relation-is-worse-than-the-rule-on-own-boxes
type: result
title: on the model's own boxes a learned contact relation calls near places contacts and does not beat the written rule
card: '159'
model: model/og4
env: environment/sprite-world
measure: percent of the learned relation's contacts that are not the simulator's, mean of 10 worlds
value: 46.6
against: 10.0
n: 10
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/contact-as-a-learned-relation
---
From [card 159](../../experiments/159-contact-as-a-learned-relation-of-places/card.md): it catches 83.4% of contacts (rule on the same boxes 74.7, 13.3% false); events F1 66.8 against 68.2 for the rule and 76.3 asked; conditions right 8 of 40 (rule 16, card 158 30). With true boxes: 98.4% caught, 0.0% false, F1 99.7, 40 of 40.
