#!/bin/sh
# Card 072, as kept (the user, 2026-10-05): trying without the relation's refits (--online 0); six decoy folds, the
# known-key control per fold (every opening in memory, no trying, no refits), then CHARTER's tiers 1 and 2.
cd "$(dirname "$0")/../../.."
export WM_REL_ENCODER=runs/070/encoder.pt
for hue in red green blue purple yellow grey; do
  WM_TRYING=1 timeout 900 bin/prun python tools/card069/run.py decoy --fold $hue --hold opening --online 0 --n 100 \
    --memory 066 --out runs/072/norefit/decoy_$hue.json > runs/072/norefit/decoy_$hue.log 2>&1
  timeout 900 bin/prun python tools/card072/control.py $hue > runs/072/norefit/control_$hue.log 2>&1
done
WM_TRYING=1 timeout 900 bin/prun python tools/card069/run.py tiers --tier 1 --online 0 --n 200 --memory 066 --out runs/072/norefit/tier1.json > runs/072/norefit/tier1.log 2>&1
WM_TRYING=1 timeout 1500 bin/prun python tools/card069/run.py tiers --tier 2 --online 0 --n 100 --memory 066 --out runs/072/norefit/tier2.json > runs/072/norefit/tier2.log 2>&1
