---
id: "124"
title: tokens from consequences — a token layer trained to predict what its parts do next under the agent's actions, with few tokens changing per event
rung: 2
serves: [P7, P4, P6, P1]
status: done
verdict: fail
arch_version: 27
date: 2026-11-10
---

# 124: tokens from consequences

Roadmap step 2 (event segmentation), rung 2. Cards 119–123 read objects
and events from a frozen encoder with no training, and each hit a limit
no threshold fixes: static objects give no signal, change gives a seed
and not an extent, the arm swallows what it holds, and what a thing
causes appears nowhere. The user (2026-11-10): "we need long training
runs where agent interaction signals and object effect signals (what
the object does / causes) influence the representation."

This card trains the token layer that sits on the frozen patches. It is
card 116.1's slot model with a different teacher. That model was
trained to reconstruct the present frame, and its slots kept the table
with each object (mean overlap 0.21, equal to k-means). Here a token is
trained on what its part does next.

Biological counterpart: object representations in the dorsal stream
are organised by how things can be acted on, not only by how they look
(Goodale and Milner 1992), and infants' object knowledge follows their
own handling of things (Needham 2000; Spelke 1990).

## 1. Question

Trained to predict the next frame from its tokens, the action and
touch, with a cost on how many tokens change, does the token layer
bind the scene's objects, and does a token carry what its object
causes? P7 (units from interaction), P4, P6 and P1 (a token means what
it lets the agent predict).

## 2. What changes

One component: what trains the token layer. The encoder stays frozen
(GOAL C1), and the slot attention, decoder and sizes are card 116.1's.

```
116.1:  tokens_t → decode → frame t's patch features                      (the present)
124:    tokens_t, action, touch → transition → decode → frame t+1's patch features   (the consequence)
        + the present, as before  + a cost on the number of tokens that change per step
```

- **Input and target:** blocks 2 and 12 of the frozen encoder, joined
  (768 numbers per patch). Card 119 showed block 2 keeps where a change
  is and block 12 loses it; card 121 showed block 12 groups by
  appearance. 256-pixel frames, 32 × 32 patches (card 122).
- **Tokens:** 12 slots of 64 numbers; a model step is 5 frames.
- **Consequence loss:** the transition's output is decoded and scored
  against the next frame before that frame is seen.
- **Sparse change:** the mean over tokens of the size of each token's
  change between steps. It is small when one token carries an event.
- **Touch:** the contact value joins the action as the transition's
  input.
- **Dropped from the proposal:** sampling batches toward events. 67% of
  4-step segments already contain a touch.

**Arms**, same data and budget: (A) card 116.1's objective, present
only; (B) this card's.

### Portability check (CHARTER rule 9; portability tests paused, AGENTS)

| Assumption | Craftium | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|---|
| The next frame depends on tokens, action and touch | holds; no touch sense, so contact is absent | the same | the same | holds | general; touch is an optional input |
| An event changes few tokens | holds for the agent's acts; night changes all | the same | mostly | holds | general, with a declared exception for scene-wide change |
| A fixed number of tokens (12) | too few for open terrain | too few (81 tiles) | varies | enough for a table top | declared setting |
| A fixed camera, 256-pixel frames | no | no | yes | fixed cameras only | declared in cards 119, 122 |

## 3. Dependencies

- Card 116.1's slot model (code reused); card 114's encoder at 448
  pixels; cards 120 and 122 for touch and resolution.
- Slots carried between frames by a learned transition, trained on
  frozen self-supervised features: SlotContrast (LITERATURE). A cost on
  how many latents change: GateL0RD (LITERATURE). Slots trained through
  an action-conditioned prediction is not an established method with
  evidence in a setting like this; that is what the card tests.
- That these features hold what the lock causes: card 114.1 (a patch
  predictor followed the lock in 97–100% of counterfactual cases).

## 4. Data check

450 training episodes of OGBench's state dataset, every fifth frame,
rendered at 256 pixels: 90,450 frames (17.8 GB, rendered in 83 s). 67%
of 4-step segments contain a touch. Validation: card 122's 100
episodes, with masks.

## 5. Feasibility gate

Run 2026-11-10, each step under 10 minutes and capped (rule 5).

1. **Does "few tokens change" prefer the true objects?**
   (`sparsity.py`, card 122's signal.) Effective number of tokens that
   change per touch, lower is sparser:

   | Grouping of the 1,024 patches | Tokens changed per event |
   |---|---|
   | True objects and background, 7 | **2.1** |
   | k-means on appearance, 7 | 2.5 |
   | A 3 × 3 grid, 9 | 3.1 |
   | Random, 7 | 5.6 |

   The pressure points at the true objects, but only a little beyond
   grouping by appearance. It cannot do the work alone.
2. **Throughput** (`train.py`). All three losses fell in every run
   (present 14.7 to 7.3, next 14.6 to 7.3, change 0.86 to 0.05).

   | Setting | Segments per second | GPU memory |
   |---|---|---|
   | Encoder in the loop, 12 segments per step | 52 | 5.6 GB |
   | Stored features, 24 per step | 68 | 11.0 GB |
   | Stored features, 24 per step, 256 of 1,024 patches scored per step | **222** | 3.6 GB |

   - Storing the features (142 GB, the user's allowance of 2026-11-10)
     gained only 1.3 times: the decoder, not the encoder, was the cost.
     Scoring a random quarter of the patches each step leaves the loss
     the same in expectation and gains 3.3 times more.
   - Features are read through a buffer of 16 episodes in memory, 4
     replaced every 100 steps, so the disk sees short sequential reads.
   - Writing the 142 GB in one pass loaded the machine (load average
     19–20 for about 15 minutes; the user reported freezes). The last 19
     episodes were written with a pause after each.
   - The evaluation (`evaluate.py`) runs all three criteria in 35 s. On
     a 3-minute model it finds no units (0.07), no self and chance on
     the lock (0.56–0.61), as expected of an untrained model.
3. **Bounds and baselines for the criteria:**
   - units: k-means on block 12 at this resolution scores arm 0.60,
     window 0.52, drawer 0.46, cube 0.06, buttons 0.06 (mean 0.29,
     card 122); card 116.3's SAM 2 teacher reached 0.50;
   - consequences: card 114.1 above; swapping a random other token
     is the trivial baseline.

**Gate result:** passed for running. Two things it does not show: that
the true grouping can be reached by training, and an upper bound from a
trained model. The change loss falling to 0.06 within 1,000 steps may
mean tokens that barely move; the main run's evaluation checks it.

## 6. Success criteria and prediction

Validation episodes, both arms:
1. **Units:** mean best-token overlap with the six objects at least
   0.40, and at least 0.26 for the cube and for each button (k-means:
   0.29 and 0.06). Arm B above arm A on the mean.
2. **Consequences:** card 114.1's counterfactual. The four frames
   before a pull are rendered with the true lock and with the lock
   flipped (button and handle repainted, nothing else), and the model
   rolls its transition 4 steps on the recorded actions. Its predicted
   change over the slide separates unlocked from locked pulls with
   balanced accuracy at least 0.9, and of the pulls it predicts right,
   at least 80% switch side when the lock is flipped. Reported: which
   token carries it (swapping only the button's token, only the
   slide's, or a random other one). Revised before any run, 2026-11-10:
   the first draft asked the button's token alone to flip the outcome,
   but the handle shows the lock too, so that test was confounded.
3. **Self:** the tokens whose prediction varies most when the action
   is varied overlap the arm with IoU at least 0.7 (card 117's bar).

**Prediction.** Honest odds are low for all three at once. Criterion 3
is the likeliest. Criterion 1 passes for the arm, drawer and window and
is the risk for the cube and buttons, which are under 1% of patches and
weigh little in a squared error. Criterion 2 needs criterion 1's button
token to exist. Arm A is expected to repeat card 116.1.

**Budget.** At 222 segments per second, 30 minutes gives 400,000
segments (61% of what card 116.1 saw before its reconstruction levelled
off), and 2 hours gives 1.6 million (2.4 times). A first pass of 30
minutes per arm can be launched here once the card is approved; longer
arms are handed to the user as commands (AGENTS: runs over 30 minutes):

```
OMP_NUM_THREADS=4 timeout 125m systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% \
  nice -n 10 bin/prun python tools/card124/train.py --minutes 120 --batch 24 --next 0 --sparse 0 --out armA.pt
OMP_NUM_THREADS=4 timeout 125m systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 -p CPUQuota=800% \
  nice -n 10 bin/prun python tools/card124/train.py --minutes 120 --batch 24 --next 1 --sparse 0.1 --out armB.pt
```

Temporary files to delete after the card: `runs/124/feat_train.npy`
(142 GB) and `runs/124/rgb_train.npy` (18 GB).

## 7. Result

First pass, 2026-11-10: 30 minutes per arm (A 16,961 steps, B 16,064
steps of 24 segments; about 400,000 segments each), then `evaluate.py`
on validation episodes. The 2-hour arms were not run.

**1. Units: fail, and arm B is below arm A.** Best-token overlap (IoU)
with each object:

| | Arm | Cube | Button 1 | Button 2 | Drawer | Window | Mean |
|---|---|---|---|---|---|---|---|
| Bar | | 0.26 | 0.26 | 0.26 | | | 0.40 |
| k-means, block 12 (card 122) | 0.60 | 0.06 | 0.06 | 0.06 | 0.46 | 0.52 | 0.29 |
| A: present only | 0.20 | **0.55** | 0.02 | 0.02 | 0.49 | 0.40 | 0.28 |
| B: next frame, few tokens change | 0.38 | 0.05 | 0.07 | 0.06 | 0.33 | 0.25 | 0.19 |

- Arm A finds the cube (0.55 against k-means 0.06; its token holds 78%
  of the cube and is 72% cube). This was not predicted: the same
  objective at 64 pixels on the last block (card 116.1) kept the table
  with every object. It splits the arm over several tokens (the best
  holds 26% of it).
- Arm B does the reverse (`diagnose.py`, 150 segments). The arm is
  more whole (64% in one token), and 67% of the cube's patches sit in
  the arm's token; the cube's best token is 6% cube.
- Neither arm has a button token (0.02–0.07).

**2. Consequences: fail.** Bars: balanced accuracy 0.9; 80% of
correctly predicted pulls switch side when the lock is flipped.

| | Drawer, A | Drawer, B | Window, A | Window, B |
|---|---|---|---|---|
| Unlocked against locked pulls, balanced accuracy | 0.63 | 0.65 | 0.52 | 0.66 |
| Switch side when the lock is flipped | 12% | 23% | 18% | 8% |
| Switch when one random token is swapped | 2% | 2% | 2% | 0% |

Arm B is above chance (0.5) but far from the bar, and no single token
carries the lock (button's token 1–3%, slide's 1–2%). Card 114.1's
patch predictor followed the lock in 97–100% of cases on the same
kind of feature, so the lock is lost in the 12 tokens, not in the
encoder.

**3. Self: fail, and the read-out is faulty.** IoU with the arm 0.11
in both arms (bar 0.7). The rule "at least 3 times the median
influence" chose 0.5 tokens per frame in arm B, so in most frames it
chose none. The number says little about the model.

**Training.** Arm B's next-frame loss equals its present loss
throughout (1.49 against 1.49 at the end), so predicting 5 frames
ahead cost nothing extra on average: most of the frame does not
change. Tokens moved less per step in B (0.18 against 0.50). Both
losses were still falling slowly at 30 minutes.

**Arm C, added 2026-11-10 with the user's approval:** the next-frame
loss with no cost on token change, 30 minutes (15,648 steps), to
separate arm B's two changes.

| | A: present | B: next + cost | C: next only | Bar |
|---|---|---|---|---|
| Units, mean of six | 0.28 | 0.19 | 0.21 | 0.40 |
| Cube | 0.55 | 0.05 | 0.14 | 0.26 |
| Arm | 0.20 | 0.38 | 0.25 | |
| Share of the cube in the arm's token | 3% | 65% | 13% | |
| Drawer: unlocked against locked, balanced accuracy | 0.63 | 0.65 | **0.83** | 0.9 |
| Drawer: switch side when the lock is flipped | 12% | 23% | **63%** | 80% |
| Window: balanced accuracy | 0.52 | 0.66 | 0.54 | 0.9 |
| Window: switch side | 18% | 8% | 5% | 80% |
| Self, IoU with the arm (tokens chosen) | 0.11 (1.2) | 0.11 (0.5) | 0.35 (2.7) | 0.7 |

Arm C still fails every criterion.

**Reading.**
- The next-frame loss alone already loses the cube as a unit (0.14):
  its patches fall in a token six times its size. The cost on token
  change then moves the cube into the arm's token (13% to 65%). One
  token changing is cheaper than two, and the gate missed this because
  it compared only groupings with the same number of groups.
- The cost also removed most of what the next-frame loss gained on the
  lock: without it the drawer's lock is nearly followed (0.83, 63%
  switch; random-token swap 11%). No single token carries it (button's
  token 1%, slide's 9%), and the window's lock is not followed at all.
- Each arm is one seed, and the arms differ by amounts (cube 0.55
  against 0.14) that one seed cannot rule out as chance.

## 8. Decision

**Revise.** No arm passes, and the 2-hour arms as designed are not
worth running. The cost on token change is dropped: it merged the cube
with the arm and hid the lock. What stands: present-only slots at 256
pixels on blocks 2 and 12 find the cube (arm A), and the next-frame
loss starts to carry the drawer's lock (arm C), but the two were not
obtained in one model. Changes for the user to choose from:

1. Weight the next-frame loss toward patches that change (card 119's
   signal). The lock and the cube are under 1% of patches and weigh
   little in a mean squared error over the frame.
2. Repeat arms A and C on a second seed before building on either.
3. Read the self as the single most action-dependent token, not by a
   threshold on the median.
