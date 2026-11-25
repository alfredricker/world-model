---
id: environment/world-of-parts
type: environment
title: World of parts (patches, objects, scenes)
links: []
---
A synthetic world built on 2026-11-20 ([card 151](../../experiments/energy-modules/151-layers-of-attractor-modules-in-a-world-of-parts/card.md), `tools/card151/layers.py`) to test layers of attractor modules where the right answer at every level is known. 16 patches of 16 numbers; 24 objects of 4 patches, half of them twins that differ from another object in one patch; 32 scenes of 4 objects, each with one of 8 output tokens. An image is a scene plus noise, with patches optionally covered; a stream changes scene every 16 ticks. Parts sit on a fixed grid and nothing moves or is acted on. A test bed for one component; whole cards are still judged on OGBench.
