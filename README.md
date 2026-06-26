# VBAD / Fabric Defect Detection — 实验脚本说明

本仓库包含 EAAI 论文 *"Learning When to Inspect at High Resolution: Visibility-Aware
Triage for Fabric Defect Detection"* 的全部实验脚本。下表把每个脚本对应到论文的章节、
表、图，并给出使用方法。

## 环境
- Python 3.10 + PyTorch 2.11 (cu128), ultralytics 8.4.65, 在 RTX 5090 上运行。
- 虚拟环境: `~/CvTest/.venv` (uv 管理)。运行前缀均为 `~/CvTest/.venv/bin/python`。
- 远程执行约定: 复杂命令写成 `/tmp/x.sh` 再 `bash` 执行 (避免 shell 引号转义)。

## 数据
- **Fabric (主数据集, 论文 Table 1)**: 公开 SciDB 数据集 (DOI 10.57760/sciencedb.17138)。
  - `data/Dataset B/` = 21 类长尾织物缺陷 (论文主体)。
  - `data/Dataset A/` = 4 类近均衡织物 (作易数据对照, §4.2 high-res harms easy data)。
  - 切分: `data/Dataset B/data_split.yaml` (train2/val2/test2，原val冻结为test)。
- **GC10-DET (对照数据集, §5 + Table tab:gc10)**: Kaggle `alex000kim/gc10det`，
  `data_gc10/yolo/data_split.yaml` (10类金属表面缺陷)。

---

## 一、数据准备脚本

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `experiments/scripts/make_testsplit.py` | 织物数据集分层重切 train2/val2/test2 (原val冻结为test) | §4.1 Dataset and protocol |
| `method/gc10_prepare.py` | GC10 VOC-XML→YOLO 格式 + 分层切分 | §5 GC10 对照 |
| `check_images.py` / `validate_yaml.py` | 数据下载后完整性校验、yaml 路径校验 | (数据准备, 不在正文) |
| `verify_gpu.py` | 验证 RTX 5090(sm_120) + cu128 torch 可用 | (环境验证) |

**用法**:
```bash
# 织物数据集重切 (一次性)
~/CvTest/.venv/bin/python experiments/scripts/make_testsplit.py --root "data/Dataset B" --train train --val val
# GC10 转换
~/CvTest/.venv/bin/python method/gc10_prepare.py
```

---

## 二、训练脚本

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `experiments/scripts/run_exp.py` | 统一训练入口 (YOLOv8/v11/RT-DETR, 任意分辨率) + head/med/tail 分组评测 | Table 2 (分辨率), Table 3 (方法对照) |
| `experiments/launch_baselines.sh` | 批量启动 baseline 矩阵 (3检测器×2数据集) | 早期诊断 baseline |
| `experiments/scripts/train_missaware.py` | miss-aware 损失重加权 (monkey-patch v8DetectionLoss) | Table 3 "miss-rate loss reweighting" |
| `experiments/scripts/train_distill.py` | 1280→640 特征蒸馏 (monkey-patch model.loss) | Table 3 "feature distillation", §5 |
| `method/train_density_mvp.py` | 密度分支训练 (纹理缺陷, 失败的尝试) | §4.3 (cheap interventions fail) |

**用法 (核心: run_exp.py)**:
```bash
PY=~/CvTest/.venv/bin/python; S=experiments/scripts/run_exp.py
B="data/Dataset B/data_split.yaml"
# 640 baseline
$PY $S --model yolov8m.pt --data "$B" --epochs 200 --imgsz 640 --batch 16 --device 2 --seed 0 --name yolov8m_B
# 高分辨率 1280 (Table 2)
$PY $S --model yolov8m.pt --data "$B" --epochs 150 --imgsz 1280 --batch 8 --device 3 --seed 0 --name v8m_B_1280
```
注意: RTX 5090 上预训练权重需经 GitHub 镜像 (gh-proxy.com / ghfast.top) 下载。

---

## 三、评测脚本

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `experiments/scripts/eval_test.py` | 在冻结 test 集评测 (支持 `--imgsz`)，输出分组 mAP/recall JSON | 全部 test 数字 |
| `experiments/scripts/summarize.py` | 汇总多个 run 的 JSON 成对比表 | Table 2/3 |
| `experiments/scripts/seedstats.py` | 多 seed mean±std | Table 2 (1280 3-seed) |
| `experiments/scripts/perclass.py` / `perclass_diff.py` | 逐类 recall / 方法前后对比 | Table 5 (per-class) |
| `method/eval_density_mvp.py` | 密度分支 test 评测 | §4.3 |

**用法**:
```bash
~/CvTest/.venv/bin/python experiments/scripts/eval_test.py \
  --weights experiments/runs/v8m_B_1280/weights/best.pt \
  --data "data/Dataset B/data_split.yaml" --imgsz 1280 --device 2 --name v8m_B_1280
```

---

## 四、诊断分析脚本 (论文的核心贡献来源)

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `experiments/scripts/check_labels.py` | 类别频次/框尺寸/极小框比例分布 | Table 1 (tiny-defect %), §4.5 |
| `experiments/scripts/confusion.py` / `confusion2.py` | 混淆矩阵 (手算 IoU匹配): 漏检 vs 混淆分解 | 早期诊断 (errors are misses not confusion) |
| `experiments/scripts/cross_detector_diag.py` / `_test.py` | 跨检测器一致性: 漏检/混淆/阈值余量 | 诊断鲁棒性 |
| `experiments/scripts/sizeanalysis.py` | recall vs 目标尺寸相关性 | 诊断 (size doesn't predict difficulty) |
| `experiments/scripts/test_diag_full.py` | test 集完整诊断 (相关性+混淆+余量) | §4 诊断 |
| `experiments/scripts/thresh.py` / `adaptive_thresh.py` | 逐类阈值余量 / 自适应阈值 | Table 3 "threshold calibration" |
| `experiments/scripts/adaptive_probe.py` | 自适应分辨率 oracle 探针 | §4.3 (adaptive resolution) |

**用法**:
```bash
~/CvTest/.venv/bin/python experiments/scripts/check_labels.py   # 标注分布
~/CvTest/.venv/bin/python experiments/scripts/test_diag_full.py # 完整诊断
```

---

## 五、VBAD 方法脚本 (论文 §3 Method + §4.4)

执行顺序: export → head → pipeline。

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `experiments/scripts/vbad_oracle.py` | VBAD image-level oracle (上界+触发率+Type-A) | §4.4, Table 4 |
| `method/vbad_export.py` | 导出 640 GAP 特征 + 4标量 + Type-V 标签 + 检测缓存 | §3.3 visibility head 输入 |
| `method/vbad_export2.py` | 导出空间特征 (v2, 对照实验) | §5 (head ceiling 讨论) |
| `method/vbad_head.py` | 训练 visibility head (GAP, 主结果) + 评测 | §4.4 VBAD learned head |
| `method/vbad_head2.py` | 空间 CNN head (v2, 更差, 说明上限) | §5 |
| `method/vbad_pipeline.py` | **正式 VBAD 推理 pipeline** (640→head→选择性1280→合并) + recall-cost | §3.2, Table 4, Fig 3 |
| `experiments/scripts/vbad_learnable.py` | 可学性验证 (真实信号 vs oracle) | §4.4 (gap to oracle) |
| `experiments/scripts/vgr_oracle.py` / `vgr_oracle2.py` | VGR patch复检 oracle (失败的尝试) | §4.3/§5 (patch loses context) |

**用法 (核心: 复现 VBAD 主结果)**:
```bash
# 1. 导出特征+Type-V标签 (需要 640 和 1280 模型已训好)
~/CvTest/.venv/bin/python method/vbad_export.py
# 2. 训练 visibility head + 评测 (输出 Table 4 的 VBAD learned head 行)
~/CvTest/.venv/bin/python method/vbad_head.py
# 3. 正式 pipeline + recall-cost 曲线 (Fig 3 数据)
~/CvTest/.venv/bin/python method/vbad_pipeline.py --tau 0.4 --device 2
```

---

## 六、GC10 对照实验脚本 (§5 适用边界)

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `method/gc10_prepare.py` | GC10 VOC→YOLO | §5 |
| `method/gc10_typev.py` | GC10 上 Type-V/Type-A 比例 | Table tab:gc10 |
| `method/gc10_applicability.py` | GC10 适用性 (high-res无益证明) | §5 "when does VBAD help" |

**用法**:
```bash
~/CvTest/.venv/bin/python method/gc10_typev.py
~/CvTest/.venv/bin/python method/gc10_applicability.py
```

---

## 七、论文图表生成脚本

| 脚本 | 作用 | 对应论文 |
|------|------|----------|
| `method/vbad_results_table.py` | 汇总 Table 2/3/4/5 全部数字 | 所有表 |
| `method/eaai_figs.py` | Fig 2 (taxonomy), Fig 3 (recall-cost), Fig 5 (ROC) | Fig 2/3/5 |
| `method/eaai_fig1.py` | Fig 1 (框架图) | Fig 1 |
| `method/eaai_fig4.py` | Fig 4 (Type-V/A 缺陷样例) | Fig 4 |
| `experiments/scripts/make_figures.py` / `_test.py` | 诊断图 (val/test 版) | 早期诊断论文 |

**用法**:
```bash
~/CvTest/.venv/bin/python method/vbad_results_table.py   # 打印所有表数字
~/CvTest/.venv/bin/python method/eaai_figs.py            # 生成 Fig 2/3/5
~/CvTest/.venv/bin/python method/eaai_fig1.py            # Fig 1
~/CvTest/.venv/bin/python method/eaai_fig4.py            # Fig 4
```

---

## 八、诚信复核脚本

| 脚本 | 作用 |
|------|------|
| `method/eaai_integrity.py` | 核验论文所有数字 vs 源 JSON/缓存 (13 OK 0 BAD) |
| `experiments/scripts/integrity_verify.py` / `final_integrity.py` | 早期诊断论文数值核验 |

```bash
~/CvTest/.venv/bin/python method/eaai_integrity.py   # 应输出 "13 OK, 0 BAD"
```

---

## 完整复现流程 (从零)

```bash
# 1. 数据准备
python experiments/scripts/make_testsplit.py --root "data/Dataset B" --train train --val val
# 2. 训练 640 + 1280 (+ 多分辨率/多seed, 见 Table 2)
python experiments/scripts/run_exp.py --model yolov8m.pt --data "data/Dataset B/data_split.yaml" --imgsz 640  --batch 16 --device 0 --name yolov8m_B_ts
python experiments/scripts/run_exp.py --model yolov8m.pt --data "data/Dataset B/data_split.yaml" --imgsz 1280 --batch 8  --device 1 --name opt1_hires1280
# 3. 评测
python experiments/scripts/eval_test.py --weights experiments/runs/yolov8m_B_ts/weights/best.pt --data "data/Dataset B/data_split.yaml" --imgsz 640 --name yolov8m_B
# 4. VBAD
python method/vbad_export.py && python method/vbad_head.py && python method/vbad_pipeline.py
# 5. 表/图
python method/vbad_results_table.py && python method/eaai_figs.py
# 6. 诚信核验
python method/eaai_integrity.py
```
