---
id: result/132-camera-view-needs-no-memory
type: result
title: with the camera's view the hidden cube needs no memory
card: '132'
model: model/og4
env: environment/ogbench-visual-scene
measure: balanced accuracy (%), is the cube inside a reopened drawer, memoryless key with perfect perception and the camera's view
value: 100.0
against: 60.0
n: 198
outcome: fail
lost_points: 0
links:
- rel: instance
  to: problem/absence-not-a-value
  weight: 1.0
  weight_basis: estimated
---
Criterion 1 of card 132 asked whether the test needs memory, with a bar of at most 60% for a memoryless key. With perfect perception and the camera's view, a memoryless key told a reopened drawer with the cube from an empty one at 100.0% balanced accuracy on 198 reopenings (40 with the cube), against the 60% bar. The cube's absence from view says where it is, because there is one cube and one place to hide it. In the narrow view (the touch alone) the same key scores 51.6%, so memory matters only when the view is partial. This narrows the problem of things out of view being forgotten: tasks 4 and 5 fail on reading, and on absence not being a value, not on memory. The weight is an estimate. Numbers are in [results.json](../../experiments/132-memory-test-bed/results.json).
