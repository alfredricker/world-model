---
id: environment/memory-world
type: environment
title: Memory world (symbolic)
links: []
---
A small symbolic world built on 2026-11-13 at the user's request ([card 136](../../experiments/136-memory-world/card.md)) to test the memory component alone. It has 17 two-state objects: 8 switches with no parents and 9 children, one child for each kind of memory (short term within an episode, carried over between episodes, a fixed start value, slow and fast unseen drift, a timer, a chance, two parents, and a parent rewired every 150 episodes). The agent sees only the object it touches, after the touch. An exact Bayesian observer over all 2^17 joint states gives the best possible answer for every touch. It is a test bed for memory only; cards on the whole agent are still judged on OGBench.
