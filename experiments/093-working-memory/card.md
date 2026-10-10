---
id: "093"
title: working memory, an intention held between steps and reconsidered on events
rung: 1
serves: [P17, P21, P12, P15]
status: draft
verdict:
arch_version: 20
date: 2026-10-09
---

# 093: working memory, an intention held between steps

**Set aside by the user (2026-10-09):** its success or failure would not
say much about the direction of the object-files and structure cards
(now 094 and 096). Kept as a draft;
its intention and triggers are taken up as step 097.2 of the
procedural-memory series (a running skill handed back on these
triggers).

Drafted 2026-10-09 from a discussion with the user: the architecture is
three kinds of memory (ARCHITECTURE.md, "Memory systems"), and working
memory comes first. Working memory holds the believed scene, the goal,
what the agent is committed to doing toward it (the intention) and what
it expects its next action to do. The user chose: "when last seen" kept
in a basic form now (to be improved for Crafter); surprise as a reason
to reconsider; one card for the intention and its triggers.

## 1. Question

Version 20 re-derives everything at every step: the full means-ends
search from the goal (card 074's orders included), and, when that finds
no plan, card 072's fallback (exploration, then up to 6 extra searches
for a guess). On tier 3 that costs 1.45 s per step, about 85% of it in
the fallback, and 15 of the 16 failures run out of the hour. If the
agent instead holds its intention between steps and reconsiders only
when an event makes it stale, does tier 3 run at least twice as many
steps per hour, with no tier worse? Rung 1; P17 (planning stays usable as
episodes grow), P21 (a held intention continued cheaply, deliberation
when needed), P12, P15.

## 2. What changes

One component: working memory, which decides whether a step deliberates
or continues. Recall, the search, walking, exploration and the guesses
are unchanged.

```
version 20, every step:   choose (search from the goal) → else fallback (explore, guess search) → action
this card, every step:    observe → check the triggers against working memory
                          any trigger:  deliberate as version 20; write the new intention and expectation
                          none:         continue: plan only the intention's current need → action
```

**Working memory** (one record per episode):
- **Belief:** the tokens and placement, as now, plus **when each place
  was last seen** (a step number per token id, −1 for never). Basic form:
  read only to tell a place seen for the first time. Crafter's moving
  creatures will need belief to age with it (a later card).
- **Goal:** as now (`pl.goal`, or "the episode ends").
- **Intention:** what the last deliberation committed to: the chain from
  the goal down to its current need (a condition bound to tokens, such as
  "face the blue door holding the blue key", or "walk to stand at a
  placement from which never-seen places come into view"), and its
  source (plan, exploration, card 060's door, or card 072's guess).
  Version 20's kept choices (`pl.commit`, `_order_keep`, `look_j`,
  trying's guess) stay as they are; they now live in this record.
- **Expectation:** the predicted next state of the chosen action
  (`pl.step`, already computed for placing): placement, front token and
  held token.

**Triggers** (any one → deliberate):
1. **Need met:** the intention's current need holds in the new state.
2. **Surprise:** the observed placement, front token or held token
   differs from the expectation.
3. **Newly seen, could act on:** a place seen for the first time holds
   the goal's token, or a token that recall says some action can change
   (pick up or toggle predicted to have an effect; card 087's properties
   where recall has no tries), cached per code. Doors, keys, balls and
   boxes qualify, floor and walls do not; a second door of a known colour
   does. Newly seen floor that opens a shorter route does not trigger
   (it costs steps, not success).
4. **Cannot continue:** planning the current need alone finds no plan.

The first step of an episode deliberates. Continuing plans only the
current need (for most steps, walking's closeness field toward its
target and one move), not the goal's chain, its orders or the fallback.

## 3. Dependencies

Cards 067 (walking), 057 and 060 (exploration, the door kept between
steps), 072 (guesses), 074 (orders, ties kept), 051 (kept choices), 050
(own tries), version 20's recall. Literature (LITERATURE.md's current
focus; Kinny and Georgeff, Rao and Georgeff in `papi`):
- execution monitoring: STRIPS plans executed with triangle tables,
  re-planning when an observation breaks a precondition (Fikes, Hart and
  Nilsson 1972);
- intentions as commitments that filter reconsideration (Bratman 1987;
  Rao and Georgeff 1995), and when to reconsider (Kinny and Georgeff
  1991: replanning every step loses when planning takes time; commitment
  with replanning on relevant change did best);
- complementary learning systems (McClelland, McNaughton and O'Reilly
  1995; Kumaran, Hassabis and McClelland 2016) for the frame.

## 4. Data check

From version 20's per-step traces (tier 2's 100 seeds; 5 tier 3 seeds
for 300 s): the share of steps on which each trigger would fire, and how
many steps the fallback ran.

## 5. Feasibility gate

**Shadow mode:** version 20 acts as now, and at each step working memory
also computes whether a trigger fires and what continuing would have
done.
- **Upper bound:** the share of tier 3 steps with no trigger s, and the
  measured cost of continuing c relative to deliberating, give the
  speed-up 1 / (1 − s + s·c). It must reach 2.
- **Safety:** on steps with no trigger, continuing gives the same action
  as deliberating on at least 90% of steps (tier 2 and tier 3). A
  difference is not always an error (ties), so the gate reports the
  differing steps by intention source.
- **Trivial baseline:** version 20 (deliberates every step).

## 6. Success criteria and prediction

Two arms on the same seeds: version 20 + working memory; and the same
plus card 092's reveal step (`WM_REVEAL=recall`), whose revision this
card answers (092's target is held like any other intention).
1. **Tier 3 speed:** steps in 20 parallel episodes of 300 s (`runs/092/speed.sh`
   load) at least 2 × version 20's 3,044.
2. **CHARTER's tiers** (per arm): tier 1 ≥ 99%; tier 2 not worse than 95%
   (sign test on differing seeds); tier 3 against version 20's 14/30.
3. **Card 092's reveal arm:** tier 3 steps within 1.5 × of the first
   arm's, so 092 can be judged on its tiers.

**Prediction.** Most tier 3 steps walk toward a held target, so s is
high (above 0.8) and continuing is cheap (one closeness field). Tier 3
reaches 2–4 × the steps and more successes, since 15 failures ran out of
time. Tier 2 stays within a seed or two. The risk is a newly seen token
that matters but that no action changes (a wall closing a route); the
gate's differing steps would show it.

**Budget.** Data check and gate: about 25 minutes. Tiers 1–2, two arms:
about 50 minutes. Tier 3, two arms: about 2 hours each, handed to the
user (runs over 30 minutes).
