---
id: "103.3"
title: the view's motion read with files explained away
rung: 1
serves: [P7, P2]
status: done
verdict: fail
arch_version: 27
date: 2026-10-12
---

# 103.3: the view's motion with files explained away

Drafted and run on 2026-10-12, in series
[103](../103-tracking-with-general-priors/card.md). It follows card
[103.1](../103.1-world-and-camera-layers/card.md): once the camera's
pixels were left out, a creature stepping on plain grass outvoted the
action's prior.

## 1. Question

Suppose the placing step and the files run together, and the world pixels a
file may explain are left out of the evidence for the view's motion. Is the
view's motion then read right on 98% of Crafter's steps? And do the files,
on these learned origins, keep identity close to what they keep on the
evaluator's? Rung 1. Serves P7 and P2.

## 2. What changes

One component: which pixels count as evidence in card 103.1's placing.

```
103.1:      S = pixels known under every candidate shift
this card:  S = the same, without world pixels within K px of a file's mask (files that have moved and been
            followed at least 3 times); card 103.2's files run in the same loop, on the origins this step chooses
            sleep moves nothing: sleeping frames are skipped, and canvas, origin and files carry through
```

- **Why widen by K:** a file may move up to K pixels. Both explanations of
  a stepping cow ("it stepped", "the view stepped") then lose the cow's
  pixels alike.
- **Arm "off":** the same loop without exclusion. It measures the loop
  itself.

### Portability check (CHARTER rule 9)

| Assumption | Crafter | ARC-AGI-3 | Robotics | Here |
|---|---|---|---|---|
| A file may explain pixels near it, not the view | yes | yes | yes | general (layers) |
| The world moves by one 2-D shift | yes | yes | no | declared, as in 103.1 |
| Sleep (no action takes effect) moves nothing | yes | — | — | declared for Crafter's adapter: frames while asleep are skipped |

## 3. Dependencies

- Card 103.1's placing and camera pixels (criterion 2 passed).
- Card 103.2's files: failed (IDF1 0.531). This card is the joint step that
  103.2's decision names; the files' weakness is a risk here (section 6).

## 4. Data check

Worker 8: 24,204 placed steps, 22% of them moves; 27,364 creature frames.

## 5. Feasibility gate

- **Upper bound:** 098.5's placing with the agent's tile declared reads
  98.1–98.5%. With the evaluator's origins the files score IDF1 0.522 on
  worker 8.
- **Baseline:** card 103.1 with the camera layer: 90.6%, and IDF1 0.303 with
  files on the robust arm's origins.
- **Smoke test (6,000 steps):**
  - Still steps rose from 83% to 90–92%.
  - Moves fell from 84% to 57–60%.
  - The files on learned origins had ID precision 0.11–0.12: false files
    exclude much of the view.

## 6. Success criteria and prediction

1. **View's motion:** at least 98% right, at least 95% at night (103.1's
   bar).
2. **Identity on these origins:** IDF1 within 0.05 of the evaluator's
   origins on worker 8 (0.522).

**Prediction:** both fail. Still steps improve, but false files exclude
evidence the moves need. Files must be reliable before they can explain
pixels away.

**Decision rules:** keep if both hold. Revise if still steps improve on
103.1 (90.8%) and moves do not fall below 103.1's (89.8%). Otherwise stop
this form.

## 7. Result

Run 2026-10-12 (`tools/card103.3/joint.py --excl on|off`, about 6 minutes
each, capped). Worker 8.

| | Right | Moves / still steps | Night | IDF1 | ID recall / precision |
|---|---|---|---|---|---|
| 103.1, camera layer (no files) | 90.6% | 89.8% / 90.8% | 72.4% | | |
| This loop, files not excluded | 82.9% | 83.1% / 82.9% | 47.2% | 0.222 | 0.417 / 0.151 |
| **This loop, files excluded** | 79.4% | 57.1% / 85.7% | 50.0% | 0.173 | 0.409 / 0.110 |

- **Both criteria fail.** Files on learned origins are mostly false (ID
  precision 0.11). Their widened masks leave out a median of 2,063 of the
  view's ~3,000 pixels per step, so moves lose their evidence (57%).
- **The loop itself costs 8 points against 103.1** (82.9% against 90.6%,
  night 47% against 72%). Its main difference from 103.1 is that the canvas
  and the origin carry through sleep, where 103.1 starts afresh; it also
  learns its parameters in two passes, not three. The likely cause: the
  world changes while the agent sleeps (night falls, creatures move), so the
  first frame after waking is matched against a stale canvas. This was not
  isolated. The declared "sleep moves nothing" holds for the agent, not for
  the world.

## 8. Decision

**Stop** this form (2026-10-12, by the card's rule: still steps and moves
are both below 103.1). Explaining pixels away needs files that are right.
False files remove the evidence the view needs. The question it was
answering, a mover's many pixels outvoting the action's prior, is taken up
by card 103.4: evidence counted per independent observation, not per pixel.
Carrying the canvas through sleep is dropped.
