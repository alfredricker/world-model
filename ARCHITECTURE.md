---
arch_version: 0
---

# Architecture

The current model. Experiment cards describe only their changes to it.
No model yet.

## Overview

One diagram (Mermaid) from observation to every loss, with tensor shapes.

## Latent state

What the state is, concretely: shape, units (tokens, slots, vectors, graph
nodes), what each part receives, what persists across steps and what resets.

## One update step

The operations applied per step, in order.

## Components

| Component | What it does | Input → output (shapes) | Source (see LITERATURE.md) | Borrowed vs changed |
|---|---|---|---|---|

## Built-in priors and supplied information

What is designed in (locality, grids, recurrence, curricula, labels used
anywhere) versus what is learned.

## Training signals

One line of maths and one plain-English sentence per loss.

## Change log

| arch_version | Date | Card | Change |
|---|---|---|---|
