---
id: paper/plato
type: paper
title: 'Intuitive physics learning in a deep-learning model inspired by developmental psychology'
cite: Piloto et al. 2022
papi: plato
links:
- rel: addresses
  to: problem/object-permanence
  basis: claimed
- rel: addresses
  to: problem/state-across-time
  basis: claimed
- rel: proposes
  to: principle/object-codes-with-per-object-memory
---
**Problem.** AI systems lack the intuitive physics that infants show: expecting that objects persist, do not pass through each other and move continuously. The paper asks what is needed to learn such expectations from visual data.

**Method.** The authors release the Physical Concepts dataset: 300,000 procedurally generated 3D training videos and violation-of-expectation probes for five concepts (continuity, object persistence, solidity, unchangeableness, directional inertia). Probes are built by splicing frames so a possible and an impossible video contain the same frames. The model, PLATO, has two parts. A perception module, an autoencoder, turns each object's image and segmentation mask into a 16-dimensional object code. A dynamics module (an interaction network feeding a per-object LSTM) takes the object codes of the current frame plus a buffer of earlier frames' codes and predicts the next frame's codes. It is trained on next-frame prediction. Surprise is the pixel error between the predicted and actual decoded frames, summed over the video. Segmentation masks and object identities across frames are given, not learned.

**Evidence.** Across five seeds, PLATO is more surprised by impossible than possible videos for all five concepts (accuracy 0.66 to 0.89 against a chance of 0.5, for example 0.891 for continuity and 0.678 for object persistence). Flat models with one vector per frame, matched either in parameters or in capacity, show surprise effects for at most three concepts, and accuracy reliably above chance for one or two. Training on 50,000 examples (28 hours of video) already gives robust effects (accuracy 0.75). Without retraining it also shows effects on three probe types from the independent ADEPT dataset, with new objects and dynamics (accuracy 0.77 to 0.97).

**Assumptions and limits.** Segmentation and tracking are given; the scenes are simple synthetic ones; the model predicts, it does not act. The authors speculate that the object structure helps as a regulariser.

**Bearing on this project.** It speaks to object-permanence and state-across-time: object-level state, kept in a buffer over past frames and updated by interaction, is its route to expecting that a hidden object still exists. One idea to borrow is a violation-of-expectation test for our own model, with surprise as the check that a hidden cube is remembered.
