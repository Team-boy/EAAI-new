#!/usr/bin/env bash
# 并行启动 baseline 矩阵: 3 模型 x 2 数据集, GPU2-7, equal-budget
set -u
PY=/home/fenghn/CvTest/.venv/bin/python
S=/home/fenghn/CvTest/experiments/scripts/run_exp.py
A="/home/fenghn/CvTest/data/Dataset A/mydata.yaml"
B="/home/fenghn/CvTest/data/Dataset B/data.yaml"
RUNS=/home/fenghn/CvTest/experiments/runs
EP=${1:-200}

echo "[$(date '+%T')] 预下载预训练权重(避免并行竞争)..."
$PY - <<'PYEOF'
from ultralytics import YOLO, RTDETR
for w in ("yolov8m.pt", "yolo11m.pt", "yolo11n.pt"):
    YOLO(w)
RTDETR("rtdetr-l.pt")
print("weights ready")
PYEOF

launch () {   # name model data gpu
  local name=$1 model=$2 data=$3 gpu=$4
  nohup $PY $S --model "$model" --data "$data" --epochs $EP --imgsz 640 --batch 16 \
        --device $gpu --seed 0 --name "$name" > $RUNS/$name.log 2>&1 &
  echo "  launched $name  GPU$gpu  pid=$!"
}

echo "[$(date '+%T')] 启动 6 个训练 (epochs=$EP):"
launch yolov8m_A yolov8m.pt  "$A" 2
launch yolov8m_B yolov8m.pt  "$B" 3
launch yolo11m_A yolo11m.pt  "$A" 4
launch yolo11m_B yolo11m.pt  "$B" 5
launch rtdetr_A  rtdetr-l.pt "$A" 6
launch rtdetr_B  rtdetr-l.pt "$B" 7
echo "[$(date '+%T')] 全部启动. 日志: $RUNS/<name>.log  结果: experiments/results/<name>.json"
