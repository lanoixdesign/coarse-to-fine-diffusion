#!/bin/bash
# Stages 2 and 3 of the long run (CPU). Resumable: re-run this script after an interruption;
# a finished stage (checkpoints_long/stageK.done) is skipped, an unfinished one resumes from
# checkpoints_long/stageK.pt. Log: /workspace/c2f_stage23.log (append).
set -u
cd /workspace/c2f-long
source /workspace/venv-c2f/bin/activate
export OMP_NUM_THREADS=8
S2_STEPS=${S2_STEPS:?}
S3_STEPS=${S3_STEPS:?}
BS=${BS:-128}
COMMON="--batch-size $BS --lr 2e-4 --ema-decay 0.999 --warmup-steps 500 --lr-schedule cosine --min-lr-ratio 0.1 \
  --color --cond label --ckpt-dir checkpoints_long --sample-dir samples_long --sample-steps 50"

run_stage() {
  local stage=$1 steps=$2; shift 2
  if [ -f checkpoints_long/stage${stage}.done ]; then echo "stage $stage already done, skipping"; return 0; fi
  local res=""
  [ -f checkpoints_long/stage${stage}.pt ] && res="--resume"
  echo "=== stage $stage: $steps steps, start $(date '+%F %T %Z') $res ==="
  python train.py --stage $stage --steps $steps $COMMON "$@" $res || { echo "stage $stage FAILED"; return 1; }
  touch checkpoints_long/stage${stage}.done
  echo "=== stage $stage done $(date '+%F %T %Z') ==="
}

echo "run_stage23.sh pid $$  S2=$S2_STEPS S3=$S3_STEPS BS=$BS"
run_stage 2 "$S2_STEPS" --cond-source gray --sample-every $((S2_STEPS / 5)) --save-every $((S2_STEPS / 10)) || exit 1
run_stage 3 "$S3_STEPS" --sample-every $((S3_STEPS / 5)) --save-every $((S3_STEPS / 10)) || exit 1
# full cascade with the default guidance (c2f.config.DEFAULT_GUIDANCE)
python make_grid.py --ckpt-dir checkpoints_long --steps 50 --seed 0 --out-dir samples_long/cascade
echo "ALL_DONE $(date '+%F %T %Z')"
