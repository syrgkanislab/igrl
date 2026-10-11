#!/usr/bin/env bash
# run_parallel.sh
# Usage: ./run_parallel.sh [python] [script.py]

start=$(date +%s.%N)

set -e                               # stop the script on the first error

PYTHON=${1:-python3}      # default interpreter
SCRIPT=${2:-experiment.py}  # default script
SCRIPTPOST=${3:-postprocess.py}

modes="1 2 3 4"

NUM_SEEDS=40  # Set this to the total number of seeds to be run
MAX_PARALLEL=20  # Set this to the desired seeds to be run in parallel

for mode in $modes; do
    case "$mode" in
    # ---------- 1 : dgp 1 experiments ----------------
    1 )  dgps="dgp1"
       archs="conv"
       latents="10"
       regweights="10.0"
       iregs="linear_hsic"
       warm_flags=("--warm_start")
       epoch_pairs=(
         "500 300 1000 100 10"
         "100  50 10000 100 10"
         "40   20 30000 256 5"
         "50   50 -1 256 5"
       ) # third part in tuple is the subsample size (-1 is full sample), batch size, patience
       ;;

    # ---------- 2 : dgp 2 experiments ----------------
    2 )  dgps="dgp2"
       archs="dense conv"
       latents="10 32"
       regweights="1.0"
       iregs="linear_hsic pairwise_hsic"
       warm_flags=(" " "--warm_start")
       epoch_pairs=("50 50 -1 256 5")
       ;;

    # ---------- 3 : dgp 3 experiments ----------------
    3 )  dgps="dgp3"
       archs="dense conv"
       latents="10 32"
       regweights="1.0"
       iregs="linear_hsic"
       warm_flags=(" ")
       epoch_pairs=("50 50 -1 256 5")
       ;;

    # ---------- 4 : dgp 4 experiments ----------------
    4 )  dgps="dgp4"
       archs="dense conv"
       latents="10 32"
       regweights="1.0"
       iregs="linear_hsic"
       warm_flags=(" ")
       epoch_pairs=("50 50 -1 256 5")
       ;;
    
    # ---------- add more cases here ------------------------------------------
    # 3 ) … ;;
    # 4 ) … ;;
    
    # ---------- anything else ------------------------------------------------
    * ) echo "Unknown MODE '$MODE'.  Valid modes: 0,1,2,…" >&2; exit 1 ;;
    esac
    # ────────────────────────────────────────────────────────────────────────────

    
    
    # ────────────────────────────────────────────────────────────────────────────
    
    for dgp in $dgps; do
      for arch in $archs; do
        for ldim in $latents; do
          for rw in $regweights; do
            for ireg in $iregs; do
              for warm in "${warm_flags[@]}"; do
                for pair in "${epoch_pairs[@]}"; do
                  read e1 e2 ss bs pt <<< "$pair"
    
                  for ((start=0; start<NUM_SEEDS; start+=MAX_PARALLEL)); do
                    for ((i=0; i<MAX_PARALLEL && (start+i)<NUM_SEEDS; i++)); do
                      it=$((start+i))
                      "$PYTHON" "$SCRIPT" $it -d $dgp -a $arch -l $ldim -rw $rw -ir $ireg $warm -e1 $e1 -e2 $e2 -ss $ss -bs $bs -pt $pt &
                    done
                    wait
                  done
    
                  "$PYTHON" "$SCRIPTPOST" 0 $((NUM_SEEDS-1)) -d $dgp -a $arch -l $ldim -rw $rw -ir $ireg $warm -e1 $e1 -e2 $e2 -ss $ss -bs $bs -pt $pt
    
                done
              done
            done
          done
        done
      done
    done
done

end=$(date +%s.%N)
elapsed=$(awk "BEGIN {print $end - $start}")
printf '\nDone in %.3f seconds\n' "$elapsed"
