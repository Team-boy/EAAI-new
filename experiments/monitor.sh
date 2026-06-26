#!/usr/bin/env bash
RUNS=/home/fenghn/CvTest/experiments/runs
RES=/home/fenghn/CvTest/experiments/results
echo "=== 训练进度 (epoch/200) ==="
for n in yolov8m_A yolov8m_B yolo11m_A yolo11m_B rtdetr_A rtdetr_B; do
  csv=$RUNS/$n/results.csv
  if [ -f "$csv" ]; then ep=$(($(wc -l < "$csv")-1)); else ep=0; fi
  d="running"; [ -f "$RES/$n.json" ] && d="DONE"
  pgrep -f "name $n" >/dev/null || [ "$d" = "DONE" ] || d="stopped?"
  printf "  %-12s epoch %3d/200  [%s]\n" "$n" "$ep" "$d"
done
echo "=== GPU 2-7 ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n "3,8p"
