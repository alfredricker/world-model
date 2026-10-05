# Agent guide

This repo builds a learned world model one tested step at a time. The rules
that govern direction changes are in [CHARTER.md](CHARTER.md); this file says
how to work day to day.

## Start of every session

Read, in order, and nothing else unless the task needs it:

1. [STATUS.md](STATUS.md): current rung, active card, next decision.
2. [GOAL.md](GOAL.md): the properties the finished system must satisfy.
3. [CHARTER.md](CHARTER.md): the ladder and the rules.
4. [ARCHITECTURE.md](ARCHITECTURE.md): the current model.
5. The "Current focus" section of [LITERATURE.md](LITERATURE.md).
6. The active card under `experiments/`.

Read [LESSONS.md](LESSONS.md) before proposing any change to the model or the
data. Use [LITERATURE.md](LITERATURE.md) and `papi` when a component's source
matters. `python tools/index.py` lists every experiment.

## Where information goes

There are no other documents. Do not create notes, reviews or summaries.

| Information | Place |
|---|---|
| What the finished system must do | `GOAL.md`; only the user changes it |
| A planned or finished experiment | `experiments/NNN-name/card.md` from [the template](experiments/_template/card.md), numbers in `results.json` |
| Where we are now | `STATUS.md`, **overwritten**, at most 40 lines |
| The current model | `ARCHITECTURE.md`; bump `arch_version` and add a change-log line when it changes |
| A durable lesson with evidence | `LESSONS.md`; prune or merge, never let it exceed three pages (the user, 2026-10-01) |
| Which paper shaped which component; the current paper focus | `LITERATURE.md` |
| What one overnight session did, card by card | `overnight-MM-DD.md`, temporary; the user deletes it once read |

## Rules

- No run longer than 10 minutes without a card whose status is `approved`
  by the user. Runs over 30 minutes are handed to the user as commands.
- A feasibility gate (upper bound + trivial baseline) passes before the
  main run. A test that cannot be passed is not an experiment.
- Change one component per experiment. Replacing the model or switching
  direction follows the CHARTER rules, and needs the user's explicit sign-off.
  The one exception is the architecture-selection screen in CHARTER.md, run
  once at arch_version 0.
- A card ends with exactly one decision: keep, revise or stop.
- A card that changes the agent reports CHARTER's three MiniGrid tiers:
  tier 1 against its threshold (≥ 99%), tiers 2 and 3 against the best
  version so far (better, no different or worse). Keep requires tier 1 to
  pass and no tier to be worse (the user, 2026-10-05).
- Do not edit `src/` while a run is using it.

## Writing

Write for a smart reader who has not seen this project. Plain English, full
sentences, terms defined on first use. Give every number with what it is
compared against. Put caveats once, where they matter, not on every line.
Use relative Markdown links, not `[[wikilinks]]`.

## Overnight sessions

The user's rules (2026-10-05):

1. At most 10 new cards in one overnight session.
2. Do not stop to wait for the user's choice of direction. Where the
   direction splits, take the most sensible option or, when the options
   are few, run them all and keep the strongest.
3. Return to [GOAL.md](GOAL.md) as often as needed to keep choices
   aligned without the user's guidance.
4. Keep `overnight-MM-DD.md`: for each card, a concise explanation and
   the decision taken.
5. Work toward the objective the user sets in STATUS.md, until the user
   wakes. The objective may be a stretch goal; most or all of the night
   goes to it.
