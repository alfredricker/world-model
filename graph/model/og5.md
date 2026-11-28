---
id: model/og5
type: model
title: 'og5: one scene model and a search over chains of touches'
links:
- rel: uses
  to: principle/plan-backward-over-conditions
- rel: uses
  to: principle/conditions-asked-of-the-effect-model
- rel: uses
  to: principle/effect-model-over-variables
- rel: uses
  to: principle/skill-as-remembered-touch
- rel: uses
  to: principle/frozen-pretrained-patch-features
- rel: uses
  to: principle/touch-brackets-events
- rel: uses
  to: principle/replay-shifted-by-read-out
- rel: uses
  to: principle/aim-read-out-from-own-contact
- rel: uses
  to: principle/groups-from-two-way-moves
- rel: uses
  to: principle/own-tries-first
- rel: uses
  to: principle/conditions-from-own-failed-try
- rel: uses
  to: principle/release-point-labels-the-object
- rel: uses
  to: principle/absence-as-a-value
- rel: has
  to: ability/scene-task1
  value: 0.987
  n: 150
  source: card 169
- rel: has
  to: ability/scene-task2
  value: 0.913
  n: 150
  source: card 169
- rel: has
  to: ability/scene-task3
  value: 0.973
  n: 150
  source: card 169
- rel: has
  to: ability/scene-task4
  value: 0.873
  n: 150
  source: card 169
- rel: has
  to: ability/scene-task5
  value: 0.78
  n: 150
  source: card 169
- rel: has
  to: ability/scene-five-task-mean
  value: 0.905
  n: 750
  source: card 169
---
og4 with its four scene readings replaced by one network trained on touch consequences ([card 168](../../experiments/168-consequence-trained-scene-world-model/card.md)) and its one-step rule replaced by a search over chains of touches through that network ([card 169](../../experiments/169-zero-heuristic-event-bfs-on-ogbench/card.md)). The current version, adopted by the user 2026-11-27.
