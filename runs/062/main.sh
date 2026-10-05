#!/usr/bin/env bash
# card 062: version 13 with the stored tries' "in view" from the believed view under the 7 x 7 occluded view
cd "$(dirname "$0")/../.."
A="--arm B --occlude --frontier --keep-look"
for S in 399 400 401 402; do
  bin/prun python tools/card062/believed.py tools/card057/partial.py runs/054/b_m0.5_$S.pt $A --out runs/062/fam_$S.json --layouts 30 > runs/062/fam_$S.out 2>&1 &
done
bin/prun python tools/card062/believed.py tools/card057/partial.py runs/054/b_m0.5_399.pt $A --out runs/062/chain_399.json @@ --chain one_door --seed 399 --n 100 > runs/062/chain_399.out 2>&1 &
wait
echo done
