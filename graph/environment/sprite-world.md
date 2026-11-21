---
id: environment/sprite-world
type: environment
title: Sprite world (a hand that carries objects)
links: []
---
A 64 by 64 pixel world built on 2026-11-19 ([card 147](../../experiments/147-binding-from-interaction-in-a-sprite-world/card.md), `tools/card147/world.py`) as the second step of the test ladder for binding. A hand that the agent moves takes an object by touching its side, carries it and lets go; 2 to 4 textured objects of width 3 to 16 lie over each other in a fixed order; some pairs look identical; a button changes colour and locks one object. The true mask of every pixel is kept for scoring. The camera is fixed and every motion is a flat shift, so motion alone says what is one object; it is a test bed for one component, and whole cards are still judged on OGBench. From [card 148](../../experiments/148-the-action-as-an-output-token/card.md) there is also a version that takes an action (the grip first, then the move) and is read as state, without the button (`tools/card148/env.py`). Card 152 (`tools/card152/world.py`) adds three behaviours (takeable, fixed, sticky) that go with an object's shape or colour, a pool of 40 individuals that return in later episodes, and still scenes of overlapping objects. [Card 153](../../experiments/153-kinds-found-by-one-energy-in-two-domains/card.md) reran card 152's worlds (seeds 1 to 10) with the kinds found by one energy.
