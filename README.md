# YOLOv5s_CBDG 陶瓷基片表面缺陷检测

基于改进 YOLOv5s 的陶瓷基片表面缺陷分类与识别（本科毕业设计）。

针对陶瓷基片表面缺陷（裂缝/针孔/斑点）检测中**小目标特征弱**与**边缘部署算力受限**两个问题，在 YOLOv5s 上组合四种改进方法，记为 **YOLOv5s_CBDG**：

| 缩写 | 方法 | 作用位置 | 实现文件 |
|---|---|---|---|
| **C** | CA 坐标注意力 (CVPR 2021) | Backbone P3/P4 C3 后 + SPPF 后 | `CoordAtt` (models/common.py) |
| **B** | BiFPN 快速归一化加权融合 (EfficientDet) | 替换 Neck 的 Concat（双向 + 同层跳连，见论文图4.9） | `BiFPN_Fusion` (models/common.py) |
| **D** | DWConv 深度卷积 | 替换 Neck 的标准 Conv | 官方 `DWConv` |
| **G** | GhostConv + C3Ghost (GhostNet) | 替换 Backbone 的 Conv + C3（含首层，见论文图5.7） | 官方 `GhostConv`/`C3Ghost` |

## 模型配置与参数量（已验证全部可构建+前向）

| 配置文件 | 参数量 | 对应论文实验 |
|---|---|---|
| `models/yolov5s.yaml` | 7.03 M | 基线（表4.1/4.2/5.3/5.4 第1组） |
| `models/yolov5s+SE.yaml` | 7.07 M | 表4.1 SE 对比 |
| `models/yolov5s+CA.yaml` | 7.06 M | 表4.1 CA 对比 / 表5.4 第2组 |
| `models/yolov5s+BiFPN.yaml` | 7.80 M | 表4.2 BiFPN 对比 |
| `models/yolov5s+CA+BiFPN.yaml` | 7.83 M | 表5.4 第3组 |
| `models/yolov5s+GhostConv+C3Ghost.yaml` | 4.13 M | 表5.3 GhostNet 对比 |
| `models/yolov5s_CBDG.yaml`（完整模型） | **4.27 M（-39%）** | 表5.4 第4组 |

## 数据集 C_DET

自建陶瓷基片缺陷数据集，3 类：`crack`（裂缝）/ `pinhole`（针孔）/ `spot`（斑点）。训练集 799 张、验证集 200 张，位于 `data/C_DET/`，配置 `data/C_DET/data.yaml`。

> 注意：pinhole 实例数明显少于其他类别，评估时需关注分类别 P/R。

## 环境配置

代码基线：[ultralytics/yolov5](https://github.com/ultralytics/yolov5) master `f44db01`（2024-09-01），运行依赖 `ultralytics` pip 包（日志/绘图）。

**GPU 训练机（RTX 5060 为 Blackwell 架构 sm_120，必须用 cu128 轮子 + Python ≥ 3.10）：**

```bash
git clone https://github.com/Nlieky/yolov5_CBDG
cd yolov5_CBDG
conda create -n yolov5 python=3.11 -y && conda activate yolov5
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt ultralytics==8.3.0 gitpython thop
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 最后一句必须输出 `True` 和 `RTX 5060` 才能开训。若 PyPI 走不通，torch 也可从 `pip install torch --index-url https://download.pytorch.org/whl/cu128` 直装（该域名与 PyPI 独立）。

## 消融实验训练（7 组，同超参保证公平）

```bash
python train.py --cfg models/yolov5s.yaml                    --name G1_baseline --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+SE.yaml                 --name T4_SE      --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+CA.yaml                 --name G2_CA      --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+BiFPN.yaml              --name T4_BiFPN   --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+CA+BiFPN.yaml           --name G3_CA_BFPN --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+GhostConv+C3Ghost.yaml  --name T5_Ghost   --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s_CBDG.yaml               --name G4_CBDG    --epochs 150 --imgsz 640 --batch-size 16 --device 0
```

- `--data`/`--hyp`/`--weights` 用 train.py 默认值即可（已指向本仓库数据与超参，预训练 `yolov5s.pt`）
- Ghost/CBDG 组预训练权重迁移率低（结构不匹配的层跳过），前期收敛慢是正常现象
- 每组结束看 `runs/train/<name>/results.png`；GPU 上每组约 10~20 分钟
- 断了续训：`python train.py --resume runs/train/<name>/weights/last.pt`

## 结果填写（训练完成后替换）

| 网络模型 | CA | BiFPN | D+G | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | 权重大小 |
|---|---|---|---|---|---|---|---|
| YOLOv5s (G1) | | | | 待填 | 待填 | 待填 | 待填 |
| +CA (G2) | ✓ | | | 待填 | 待填 | 待填 | 待填 |
| +CA+BiFPN (G3) | ✓ | ✓ | | 待填 | 待填 | 待填 | 待填 |
| YOLOv5s_CBDG (G4) | ✓ | ✓ | ✓ | 待填 | 待填 | 待填 | 待填 |

> 论文旧数据（mAP 68.9→79.4）来自未真实训练的模型，全部以本次实测替换。分类别 AP 取自各组 `results.csv` / 验证输出；GFLOPs 在训练开始时自动打印（需 thop）。

## 推理与导出

```bash
python detect.py --weights runs/train/G4_CBDG/weights/best.pt --source data/C_DET/valid/images --conf 0.25
python val.py --weights runs/train/G4_CBDG/weights/best.pt --data data/C_DET/data.yaml
python export.py --weights runs/train/G4_CBDG/weights/best.pt --include onnx
```

## 项目结构

```
├── train.py / val.py / detect.py / export.py     # 入口（基线官方代码）
├── models/
│   ├── common.py                                 # + SEAttention / CoordAtt / BiFPN_Fusion
│   ├── yolo.py                                   # parse_model 注册新模块
│   ├── yolov5s*.yaml                             # 7 个消融配置
├── data/C_DET/                                   # 数据集 799+200
├── data/hyps/                                    # 超参
└── utils/                                        # 官方 f44db01 工具库
```

## 致谢

- [ultralytics/yolov5](https://github.com/ultralytics/yolov5)（AGPL-3.0）
- Hou Q, et al. **Coordinate Attention for Efficient Mobile Network Design.** CVPR 2021.
- Tan M, et al. **EfficientDet: Scalable and Efficient Object Detection.** CVPR 2020.
- Han K, et al. **GhostNet: More Features from Cheap Operations.** CVPR 2020.
- Hu J, et al. **Squeeze-and-Excitation Networks.** CVPR 2018.

## 许可

继承 Ultralytics YOLOv5 的 [AGPL-3.0](LICENSE)；数据集 `data/C_DET/` 仅用于学术研究。
