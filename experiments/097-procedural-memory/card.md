---
id: "097"
title: procedural memory, skills run without recall or search (a series, 097.x)
rung: 1
serves: [P21, P9, P17, P12]
status: draft
verdict:
arch_version: 20
date: 2026-10-09
---

# 097: procedural memory (the series)

Set up 2026-10-09 with the user: a fourth memory beside working, event
and structure memory (ARCHITECTURE.md, "Memory systems"), so that
efficient action (no re-planning every step, no recall over everything)
has a home. This card states the series; each step is its own card,
097.1, 097.2, …, with one component, a gate and a decision. This card
has no run of its own; it is decided when the series is. The series
starts after cards 094 (object files and a layout) and 096 (structure): its skills read card
094's belief, and building them first would mean rebuilding them (the
user, 2026-10-09).

## 1. Question

Version 20 acts only by deliberation: every step recalls over stored
tries and searches backward from the goal (System 2). People run
practised skills without deliberating (System 1) and deliberate when a
skill fails or the situation is new. Can the agent learn skills from its
own experience, run them without recall or search, and hand control
back to deliberation when they stop applying, so that most steps cost
little and nothing is lost on the tiers? P21 (the network maps state and
goal straight to an action), P9 (reusable skills), P17, P12.

## 2. The frame

- **A skill** is a learned mapping from the scene and a target condition
  to the next action, with the conditions under which it applies and the
  condition that ends it: an option (Sutton, Precup and Singh 1999),
  stated in this architecture's conditions.
- **Which one answers** follows CHARTER's rule "recall and networks, each
  where it predicts better": the skill acts where it has predicted the
  agent's own held-out experience better than deliberation in situations
  like the present one; elsewhere deliberation does (as in arbitration
  between habits and planning by their uncertainty, Daw, Niv and Dayan
  2005).
- **Handing back:** a running skill is not re-planned each step. It hands
  control back when its ending condition holds, when it is surprised, when
  a newly seen token could be acted on, or when it cannot continue (card
  093's four triggers, which this series takes up).
- **Where skills come from:** first from stored experience, trained
  offline (097.1); later compiled from goal trees by card 096's
  consolidation (as DreamCoder's sleep compiles reusable pieces).

## 3. The series

| Step | Skill | Source | Status |
|---|---|---|---|
| [097.1](../097.1-motion-as-learned-skill/card.md) | Walking: reach a placement facing a token, with waypoint conditions where it is out of reach | Card 075 (drafted 2026-10-06), adapted | Draft |
| 097.2 | Running a skill without re-planning: card 093's held intention and triggers, applied to a skill in progress | Card 093 | Not drafted |
| 097.3 | Compiled chains: a sequence of conditions that reached goals before (get the key, open the door) run as one skill | Card 096's goal trees | Not drafted; after 096's review |

## 4. Literature (reviewed before 097.1 is approved)

Squire's division of long-term memory into declarative and procedural;
options (Sutton, Precup and Singh 1999); habits and goal-directed control
(Daw, Niv and Dayan 2005); `dreamcoder-2` (Ellis et al. 2021); and card
075's papers (hindsight relabelling `1707_01495`, quasimetric distances
`2211_15120` and `2509_20478`, `hiql`, waypoints from a learned distance
`1906_05253`).
