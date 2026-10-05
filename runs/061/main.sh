#!/usr/bin/env bash
# card 061: the move model and undraw learned from the 7 x 7 occluded view; version 12's agent acting on it
cd "$(dirname "$0")/../.."
A="--arm B --occlude --frontier --keep-look"
for S in 399 400 401 402; do
  bin/prun python tools/card061/moves.py tools/card057/partial.py runs/054/b_m0.5_$S.pt $A --out runs/061/fam_$S.json --layouts 30 > runs/061/fam_$S.out 2>&1 &
done
bin/prun python tools/card061/moves.py tools/card057/partial.py runs/054/b_m0.5_399.pt $A --out runs/061/chain_399.json @@ --chain one_door --seed 399 --n 100 > runs/061/chain_399.out 2>&1 &
wait
echo done
