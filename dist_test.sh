#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OPEN_PORT=$4
PORT=${PORT:-$OPEN_PORT}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

if [ ! -d "result" ]; then
  mkdir result
fi

# r=10
# echo "Running test with r=$r"
# out="result/$r.txt"
# CONFIG="configs/diff_r_test_speed/tome-vit-adapter-t-3x_$r.py"
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     -- $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5} >> $out
for r in $(seq 100 100 1000); do
    echo "Running test with r=$r"
    out="result/$r.txt"
    CONFIG="configs/diff_r_test_speed/tome-vit-adapter-t-3x_$r.py"
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        -- $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5} >> $out
done

