---
id: principle/object-codes-with-per-object-memory
type: principle
title: Object codes with a per-object memory
status: tested
links:
- rel: addresses
  to: problem/object-permanence
  basis: claimed
---
Represent the scene as one code per object, keep each object's recent history in a buffer, and predict the next codes through interactions between objects. In PLATO (Piloto et al. 2022) this gave expectations that objects persist and do not pass through each other, where one vector per frame did not. It was given the segmentation and tracking. [Card 134.1](../../experiments/134.1-recall-keyed-per-object/card.md) tested it in a memory test bed with the objects given: recall keyed on each object's own record, plus the other records the store shows to matter, reaches 90.3% balanced accuracy where card 134's flat key reached 57.9, and 95.3 where the lock was seen (flat 88.5), but needs about 400 stored episodes (58.4 with 100).
