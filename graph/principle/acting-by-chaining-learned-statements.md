---
id: principle/acting-by-chaining-learned-statements
type: principle
title: The agent acts by chaining learned statements backward from a goal, walking by closeness propagated over its learned move effect
status: tested
links:
- rel: addresses
  to: problem/relations-between-variables
  basis: claimed
---
A goal is an event seen at example success steps. To cause an effect the agent reads its statement from the transition energy ([events as latent state changes](events-as-latent-state-changes.md)): an unmet condition on what is out of view becomes the goal of the effect that makes it true; then the agent goes to a place the statement allows, by Dijkstra over what its learned move effect says each action does on the believed map; a blocked walk asks first for the known effect that unblocks it; a thing is put down only once nothing stands between the agent and the place. Tested in [card 160](../../experiments/energy-modules/160-minigrid-goals-against-ordinary-baselines/card.md): told MiniGrid's rules it solves five tasks 100 of 100; on the learned energy it solves DoorKey in sizes never experienced (99%) and fails the three tasks whose doors are not yellow. Kept; the learned energy under it is to be revised.
