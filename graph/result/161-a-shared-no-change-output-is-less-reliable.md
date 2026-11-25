---
id: result/161-a-shared-no-change-output-is-less-reliable
type: result
title: one shared output for "nothing changes" is best in one experience and worse in the other two
card: '161'
model: model/og4
env: environment/minigrid
measure: percent of 100 fresh layouts solved, mean of five tasks and of 3 experiences
value: 77.5
against: 86.8
n: 3
outcome: fail
lost_points: 0
links:
- rel: tested
  to: principle/one-shared-output-for-no-change
---
Against: the same agent with card 160's perceptron. By experience 89.4, 74.4, 68.8 (the perceptron 87.6, 86.6, 86.2). Its answer to "will the pickup happen" is right in 98.1, 87.3 and 68.7% of cases. The gate, run on the first experience alone, had shown it ahead (38 and 37 of 40 against 30 and 33).
