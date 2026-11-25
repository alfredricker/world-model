---
id: result/159-objects-bound-by-appearance-are-partial
type: result
title: the box of an object bound by appearance and not yet moved is the true box in about half of the cases, and the partial pattern is never corrected
card: '159'
model: model/og4
env: environment/sprite-world
measure: percent of objects held as a still group whose box is exact at the first step (development world by shape)
value: 42.9
against: 85.2
n: 49
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/contact-as-a-learned-relation
---
From [card 159](../../experiments/energy-modules/159-contact-as-a-learned-relation-of-places/card.md), development worlds: exact for 85.2 and 94.9% of objects stored after they moved and recognised later, 42.9 to 63.9% for objects bound by the appearance couplings (median 0.71 to 0.89 of the object's pixels). Card 156 counted an object bound at an overlap of 0.5. When such an object later moves the moved group is taken as known and the partial pattern stays. This, not the contact rule, is behind the missed contacts of cards 157 and 158. Checks after the card: the partial patterns were stored while the appearance couplings were young (overlap 0.41 to 0.49 in episodes 10 to 19) and are reused; rewriting on motion would touch 5 of 621 sequences; the predicted coupling, added at the size of a whole episode of motion evidence, also spoils groups bound by motion (0.95 to 0.51).
