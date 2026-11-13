---
id: problem/prior-over-unseen-state
type: problem
title: What has not been seen yet in the episode cannot be recalled
parent: problem/state-across-time
links: []
---
In 223 of the 2,226 test pulls the lock had not yet been seen in the episode, and 52 of the 222 failed pulls are among them. No memory of the episode can answer those: every key scores about 63% there, against 98.2% with the true values. This calls for a prior over likely states, not for better recall ([card 133.1](../../experiments/133.1-key-prior-corrected-by-evidence/card.md)).
