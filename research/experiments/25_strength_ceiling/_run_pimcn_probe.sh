#!/bin/zsh
# Exploitation probe: does MORE determinization in play (better/closer-to-optimal
# play) help or HURT vs imperfect defenders? If soloist GP falls as PIMC_N rises,
# it confirms the play-ceiling finding (imperfect play exploits imperfect defenders).
cd /Users/milansimity/Cuccok/kodok/oldtawer
LOG=experiments/25_strength_ceiling/run_pimcn_probe.log
echo "=== PIMC_N play probe (FLOOR=0.7, N=150) START $(date) ===" > $LOG
for pn in 4 8 16 32; do
  echo ">>> PIMC_N=$pn" >> $LOG
  FLOOR=0.7 N=150 SCORER=pimc PIMC_N=$pn DEBIAS_PCTL=0.80 CALIBRATE=1 python3 experiments/24_bidding_loop/harness.py 2>&1 \
    | grep -vE "^ *[0-9]+/150" | grep -E "METRIC|NON-FLOOR|GP/seat" >> $LOG
  echo >> $LOG
done
echo "=== done $(date) ===" >> $LOG
