#!/bin/sh
# Card 072: card 070's configuration with trying (WM_TRYING=1): six decoy folds, then CHARTER's tiers 1 and 2.
cd "$(dirname "$0")/../.."
export WM_REL_ENCODER=runs/070/encoder.pt WM_TRYING=1
for hue in red green blue purple yellow grey; do
  timeout 900 bin/prun python tools/card069/run.py decoy --fold $hue --hold opening --online 1 --n 100 --memory 066 \
    --out runs/072/decoy_$hue.json > runs/072/decoy_$hue.log 2>&1
done
timeout 900 bin/prun python tools/card069/run.py tiers --tier 1 --online 1 --n 200 --memory 066 --out runs/072/tier1.json > runs/072/tier1.log 2>&1
timeout 1500 bin/prun python tools/card069/run.py tiers --tier 2 --online 1 --n 100 --memory 066 --out runs/072/tier2.json > runs/072/tier2.log 2>&1
