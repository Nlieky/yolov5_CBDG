# YOLOv5s-Ghost 轻量化混凝土表面缺陷检测

基于轻量化改进 YOLOv5s 的混凝土表面缺陷检测系统（本科毕业设计）。

针对混凝土表面缺陷检测在边缘设备上部署时算力与存储受限的问题，在 YOLOv5s 基础上引入 **GhostNet** 的幽灵卷积（GhostConv）与 C3Ghost 模块替换原网络中的标准卷积和 C3 模块，在保持检测精度的前提下显著压缩模型规模。

## 改进方案

| 模型 | 参数量 | 压缩率 | 配置文件 |
|---|---|---|---|
| 原版 YOLOv5s | 7.03 M | — | `models/yolov5s.yaml` |
| **YOLOv5s-Ghost（本项目）** | **4.13 M** | **↓ 41%** | `models/yolov5s+GhostConv+C3Ghost.yaml` |

具体改动：

- **Backbone**：下采样卷积 `Conv → GhostConv`，特征提取模块 `C3 → C3Ghost`
- **Neck/Head**：`C3 → C3Ghost`，1×1 / 3×3 卷积保留标准 `Conv` 以保障检测精度
- 首层卷积保留标准 `Conv` 并显式指定 `p=2`（`[64, 6, 2, 2]`），保证偶数倍下采样对齐

## 数据集 C_DET

混凝土表面缺陷数据集，3 个类别，位于 `data/C_DET/`：

| 类别 | 中文 | 说明 |
|---|---|---|
| crack | 裂缝 | 主要类别 |
| pinhole | 针孔 | 样本较少（类别不平衡） |
| spot | 斑点 | — |

- 训练集 799 张 / 验证集 200 张，YOLO txt 标注格式
- 数据配置：`data/C_DET/data.yaml`

## 环境配置

- Python ≥ 3.8，PyTorch ≥ 1.8（CPU / GPU 均可）
- 代码基线：[ultralytics/yolov5](https://github.com/ultralytics/yolov5) master 分支 `f44db01`（2024-09-01）

```bash
pip install -r requirements.txt
pip install ultralytics==8.3.0 gitpython
```

> 注：本项目代码运行时依赖 `ultralytics` 包（日志与绘图工具）；若 PyPI 访问受限，可从 GitHub 源安装 `pip install --no-deps git+https://github.com/ultralytics/ultralytics@v8.3.0`。

## 快速开始

```bash
# 训练（有 GPU 的机器建议 --batch-size 16）
python train.py --data data/C_DET/data.yaml --cfg models/yolov5s+GhostConv+C3Ghost.yaml \
    --weights yolov5s.pt --epochs 50 --imgsz 640 --batch-size 16

# 基线对比（消融实验：原版 YOLOv5s）
python train.py --data data/C_DET/data.yaml --cfg models/yolov5s.yaml \
    --weights yolov5s.pt --epochs 50 --imgsz 640 --batch-size 16

# 推理
python detect.py --weights runs/train/exp/weights/best.pt --source {img/dir/video/camera}

# 验证
python val.py --weights runs/train/exp/weights/best.pt --data data/C_DET/data.yaml

# 导出（ONNX / TensorRT / TFLite 等）
python export.py --weights runs/train/exp/weights/best.pt --include onnx
```

训练产物保存在 `runs/train/`（损失与指标曲线见 `results.png`），已被 `.gitignore` 排除，不入库。

## 实验结果

<!-- 正式训练完成后填写 -->

| 模型 | 参数量 | mAP@0.5 | mAP@0.5:0.95 | P | R |
|---|---|---|---|---|---|
| YOLOv5s | 7.03 M | 待填 | 待填 | 待填 | 待填 |
| YOLOv5s-Ghost | 4.13 M | 待填 | 待填 | 待填 | 待填 |

> 训练流水线已在 CPU 环境完成 1 epoch 冒烟验证（数据加载、前向/反向、验证、权重保存全部正常）。注意 pinhole 类实例较少，评估时需关注分类别的 P/R 指标。

## 项目结构

```
├── train.py / val.py / detect.py / export.py   # 训练/验证/推理/导出入口
├── models/
│   ├── yolov5s+GhostConv+C3Ghost.yaml          # 轻量化改进配置
│   ├── yolov5s.yaml 等                         # 官方基线配置
│   └── common.py / yolo.py ...                 # 模型定义（与官方 f44db01 一致）
├── data/
│   ├── C_DET/                                  # 数据集（799+200 张）
│   └── hyps/                                   # 超参配置
├── utils/                                      # 工具库（官方 f44db01）
├── ViT.py                                      # ViT 参考实现（实验草稿，未接入）
└── yolov5s.pt                                  # COCO 预训练权重
```

## 致谢

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)（AGPL-3.0）
- Han K, et al. **GhostNet: More Features from Cheap Operations.** CVPR 2020.

## 许可

继承 Ultralytics YOLOv5 的 [AGPL-3.0](LICENSE) 许可证；数据集 `data/C_DET/` 仅用于学术研究。
