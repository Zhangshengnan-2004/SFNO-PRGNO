#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${SFNO_PYTHON:-/home/zwu/miniconda3/envs/sfno/bin/python}"
EXPERIMENT_DIR="Checkpoint/v1_x8_seed42"
if [[ -d "$EXPERIMENT_DIR" ]] && [[ -n "$(ls -A "$EXPERIMENT_DIR")" ]]; then
    echo "Experiment directory is not empty: $EXPERIMENT_DIR. Refusing to overwrite."
    exit 1
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
mkdir -p "$EXPERIMENT_DIR"
"$PYTHON_BIN" -u Train.py \
    --model sfno_prgno --guidance-mode physics \
    --data_path ./Data/Train --epochs 200 --batch_size 1 \
    --patch_size 64 --lr 0.0001 --sf 8 --gpu 0 \
    --trainset_num 1000 --num_workers 2 --seed 42 \
    --save_dir "$EXPERIMENT_DIR" > "$EXPERIMENT_DIR/train.log" 2>&1
"$PYTHON_BIN" -u Test.py \
    --model sfno_prgno --guidance-mode physics \
    --data_path ./Data/Test --checkpoint "$EXPERIMENT_DIR/last.pth" \
    --sf 8 --gpu 0 --seed 1 --report "$EXPERIMENT_DIR/metrics.json" \
    > "$EXPERIMENT_DIR/test.log" 2>&1
cat "$EXPERIMENT_DIR/test.log"
