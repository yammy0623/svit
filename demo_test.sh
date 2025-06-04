#!/usr/bin/env bash
r=10
echo "Running test with --r=$r"
CONFIG="configs/diff_r_test/tome-vit-adapter-t-3x_$r.py"
echo "config $CONFIG"
python selector_demo.py data/coco/val2017/000000046252.jpg $CONFIG pretrained/vit-adapter-t-3x.pth $r


for r in $(seq 100 100 1000); do
    echo "Running test with --r=$r"
    CONFIG="configs/diff_r_test_demo/tome-vit-adapter-t-3x_$r.py"
    echo "config $CONFIG"
    python selector_demo.py data/coco/val2017/000000046252.jpg $CONFIG pretrained/vit-adapter-t-3x.pth $r
done
