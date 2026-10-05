#!/usr/bin/env bash
# card 057: arm B (frontier exploration) on seeds 400-402, familiar worlds and chained rooms with one door
cd "$(dirname "$0")/../.."
for S in 400 401 402; do
  W=runs/054/b_m0.5_$S.pt; P="bin/prun python tools/card057/partial.py $W"
  $P --arm B --out runs/057/fam_B_$S.json --layouts 30 > runs/057/fam_B_$S.out 2>&1 &
  $P --arm B --out runs/057/chain_B_$S.json @@ --chain one_door --seed $S --n 100 > runs/057/chain_B_$S.out 2>&1 &
done
wait
echo done
