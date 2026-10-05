#!/usr/bin/env bash
# card 057: the 7 x 7 view; arms A (random when there is no chain) and B (frontier exploration);
# familiar worlds (30 layouts each) and chained rooms with one door (100 layouts). Usage: runs/057/main.sh SEED
cd "$(dirname "$0")/../.."
S=${1:-399}; W=runs/054/b_m0.5_$S.pt
P="bin/prun python tools/card057/partial.py $W"
for A in A B; do
  $P --arm $A --out runs/057/fam_${A}_$S.json --layouts 30 > runs/057/fam_${A}_$S.out 2>&1 &
  $P --arm $A --out runs/057/chain_${A}_$S.json @@ --chain one_door --seed $S --n 100 > runs/057/chain_${A}_$S.out 2>&1 &
done
wait
echo done
