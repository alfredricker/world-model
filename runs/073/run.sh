#!/bin/sh
# Card 073: version 16 with needs ordered by conflicting states (WM_ORDER=1). Criterion 1: the decoy world with the
# key known, each fold's door hue (and version 16 alone for reference); criterion 3: card 072's folds with trying;
# criterion 2: CHARTER's tiers 1 and 2.
cd "$(dirname "$0")/../.."
export WM_REL_ENCODER=runs/070/encoder.pt WM_TRYING=1
for hue in red green blue purple yellow grey; do
  WM_ORDER=1 timeout 900 bin/prun python tools/card073/known.py $hue runs/073/known_$hue.json > runs/073/known_$hue.log 2>&1
  timeout 900 bin/prun python tools/card073/known.py $hue runs/073/known_v16_$hue.json > runs/073/known_v16_$hue.log 2>&1
  WM_ORDER=1 timeout 900 bin/prun python tools/card069/run.py decoy --fold $hue --hold opening --online 0 --n 100 \
    --memory 066 --out runs/073/decoy_$hue.json > runs/073/decoy_$hue.log 2>&1
done
WM_ORDER=1 timeout 900 bin/prun python tools/card069/run.py tiers --tier 1 --online 0 --n 200 --memory 066 --out runs/073/tier1.json > runs/073/tier1.log 2>&1
WM_ORDER=1 timeout 1500 bin/prun python tools/card069/run.py tiers --tier 2 --online 0 --n 100 --memory 066 --out runs/073/tier2.json > runs/073/tier2.log 2>&1
