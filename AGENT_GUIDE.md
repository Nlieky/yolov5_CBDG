# YOLOv5s_CBDG 训练机执行手册（给 5060 机器上的智能体）

> 本文档是完整交接手册。按顺序执行各阶段，每阶段有**验收标准**，不达标不进入下一阶段。

---

## 0. 你的角色与红线（先读这里）

你负责在一台 **RTX 5060** 机器上完成本毕设项目的全部训练实验。项目代码、数据、工具已全部就绪并验证。

**绝对红线：**
1. **所有写进结果的数字必须来自真实运行的输出**（`results.csv` / 验证日志），禁止估算、编造、复制其他实验的数字。上一轮数据就是因为编造而全部作废的。
2. **不得修改 `data/C_DET/valid/` 下的任何文件**——验证集是只读的，这是指标公信力的基础。
3. 7 组消融实验的**超参数、epochs、imgsz、batch、随机种子必须完全一致**（用下面给定的命令，不要自作主张调参），否则对比不公平。
4. 每完成一个阶段：`git add -A && git commit && git push`。仓库是 `https://github.com/Nlieky/yolov5_CBDG`。
5. 遇到本手册没覆盖的问题：解决后把现象和方案追加到本文档第 8 节。

**给用户的报告方式**：每阶段结束输出一段简明汇总（各 run 的 mAP@0.5 / mAP@0.5:0.95 / 参数量 / GFLOPs / 耗时），用户要拿这些数字填毕业论文。

---

## 1. 实验背景（30 秒读完）

- **课题**：基于改进 YOLOv5s 的**陶瓷基片**表面缺陷检测（本科毕设，三峡大学）。
- **模型**：YOLOv5s_CBDG = 基线 YOLOv5s 上组合 4 种改进——**C**A 坐标注意力（Backbone P3/P4/SPPF 后）+ **B**iFPN 加权特征融合（替换 Neck 的 Concat）+ **D**WConv（替换 Neck 的 Conv）+ **G**hostNet（Backbone 的 Conv→GhostConv、C3→C3Ghost）。
- **数据集 C_DET**：3 类缺陷 crack（裂缝）/ pinhole（针孔）/ spot（斑点）。训练集 792 张、验证集 200 张，250×250 二值掩膜风格图，多边形格式标注（代码原生支持）。**刚完成清洗**：修复了乱码标签、train/valid 泄漏、退化框，现在数据是干净的。
- **为什么要重训**：论文旧数据（mAP 68.9→79.4）经查证是编造的——当时只真实训练过 Ghost 单项模型。本轮要产出**全部消融组的真实数据**。
- **代码基线**：ultralytics/yolov5 master `f44db01`（2024-09-01），运行依赖 `ultralytics` pip 包。已验证 7 个模型配置在 CPU 上构建+前向全部通过。

---

## 2. 机器与素材清单

| 项目 | 说明 |
|---|---|
| GPU | RTX 5060（Blackwell，sm_120 架构——**必须 cu128 版 torch，老版本不认这卡**） |
| 代码+数据 | `git clone https://github.com/Nlieky/yolov5_CBDG` |
| 未标注图片池 | 用户会用 U 盘拷来 `dataset/` 文件夹（5461 张，约 47MB），放到仓库**同级目录**（即 `../dataset`）。若未到货，先跳过阶段三，直接做阶段二和五。 |

---

## 3. 阶段一：环境搭建与验证

```bash
git clone https://github.com/Nlieky/yolov5_CBDG
cd yolov5_CBDG
conda create -n yolov5 python=3.11 -y
conda activate yolov5
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt ultralytics==8.3.0 gitpython thop
```

> 若 `requirements.txt` 安装报冲突，可去掉版本下限逐个装核心包：`pip install opencv-python pyyaml pandas matplotlib seaborn tqdm psutil gitpython thop ultralytics==8.3.0`。

**验收（三条全过才算通过）：**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望: 2.7.x+cu128 True NVIDIA GeForce RTX 5060

python -c "
import sys; sys.path.insert(0,'.')
from models.yolo import Model
m = Model('models/yolov5s_CBDG.yaml', ch=3, nc=3)
print('CBDG 构建OK:', sum(p.numel() for p in m.parameters())/1e6, 'M 参数')"

python train.py --epochs 1 --imgsz 320 --batch-size 8 --device 0 --name smoke --exist-ok
# 期望: 跑完 1 epoch 无报错, runs/train/smoke/ 生成 weights
```

冒烟测试通过后删掉它：`rm -rf runs/train/smoke`

---

## 4. 阶段二：7 组消融训练（核心任务）

在仓库根目录依次执行（**参数一个都不要改**，`--data --hyp --weights` 用 train.py 默认值即可）：

```bash
python train.py --cfg models/yolov5s.yaml                   --name G1_baseline --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+SE.yaml                --name T4_SE       --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+CA.yaml                --name G2_CA       --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+BiFPN.yaml             --name T4_BiFPN    --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+CA+BiFPN.yaml          --name G3_CA_BFPN  --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s+GhostConv+C3Ghost.yaml --name T5_Ghost    --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s_CBDG.yaml              --name G4_CBDG     --epochs 150 --imgsz 640 --batch-size 16 --device 0
```

每组预计 10~25 分钟，总共 2~3 小时。建议写成循环脚本一次跑完，每跑完一组立即记录数字。

**每个 run 结束后立即采集（以 G4 为例）：**

```bash
# 总体指标: results.csv 中 mAP_0.5 最大的那一行就是 best
python -c "
import pandas as pd
df = pd.read_csv('runs/train/G4_CBDG/results.csv')
df.columns = [c.strip() for c in df.columns]
best = df.loc[df['metrics/mAP_0.5'].idxmax()]
print('best epoch:', int(best['epoch']), '| P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f' % (
    best['metrics/precision'], best['metrics/recall'], best['metrics/mAP_0.5'], best['metrics/mAP_0.5:0.95']))"

# 分类别 AP (论文需要 crack/pinhole/spot 各自的 AP)
python val.py --weights runs/train/G4_CBDG/weights/best.pt --data data/C_DET/data.yaml --device 0

# 模型大小 (论文"模型大小"列)
ls -la runs/train/G4_CBDG/weights/best.pt
```

GFLOPs 在每组训练日志开头一行 `Model summary: ... layers, ... parameters, ... GFLOPs`，注意保存（`tee` 或截图）。

**验收标准：**
- 7 组全部跑完，`results.csv` 完整
- 训练曲线正常收敛（`results.png` 无 NaN、无爆炸）
- 预期合理趋势（不强求）：G4 参数量 ≈ 4.27M 明显小于 G1 的 7.03M；mAP 若 G4 没超过 G1，如实记录并在报告里说明（Ghost 换精度换速度是正常 trade-off，论文照实写）
- pinhole 类 AP 大概率最低（train 只有 169 个实例），**如实记录**，这是论文讨论部分的真实素材

**每组跑完 commit 一次：** `git add -A && git commit -m "训练完成: G4_CBDG" && git push`

---

## 5. 阶段三：半自动标注扩充数据（依赖图片池到货）

前提：阶段二已完成，用**最好的那个 best.pt**（一般 G1 或 G4）。

```bash
# 1. 预标注 ../dataset 池子的 5461 张 (其中 458 张与训练集重复会自然检出, 无妨)
python autolabel.py --weights runs/train/G1_baseline/weights/best.pt \
    --source ../dataset --conf 0.4 --outdir autolabel_out --device 0

# 2. 把图片目录和预标注整理成 labelImg 可直接打开的结构
mkdir -p autolabel_out/workspace/images autolabel_out/workspace/labels
cp ../dataset/*.jpg autolabel_out/workspace/images/
cp autolabel_out/labels/*.txt autolabel_out/workspace/labels/
echo -e "crack\npinhole\nspot" > autolabel_out/workspace/classes.txt
```

**3. 人工复核（这一步是用户自己做，你把准备好并把下面这段话告诉用户）：**

> 请打开 labelImg（`pip install labelImg && labelImg autolabel_out/workspace/images autolabel_out/workspace/labels autolabel_out/workspace/classes.txt`，切换到 YOLO 格式保存）。按 `autolabel_out/review.csv` 的优先级顺序校对：优先级 0（无检出）先看，其次优先级 1（低置信），**优先挑类别分布里有 pinhole/spot 的图**（这两类样本最稀缺）。校对完的图保持在 workspace 里不动即可。建议校对 500~1000 张或至少 1 小时。

**4. 合并已复核数据（用户说校对完成后执行）：**

```bash
python - << 'EOF'
import glob, os, shutil, hashlib
# 只合并用户复核时间戳晚于 autolabel 生成的标签? 简化: 用户确认后全量合并
# 先剔除与现有 train/valid 内容重复的 458 张
have = set()
for split in ['train','valid']:
    for f in glob.glob(f'data/C_DET/{split}/images/*.jpg'):
        have.add(hashlib.md5(open(f,'rb').read()).hexdigest())
n = 0
for img in glob.glob('autolabel_out/workspace/images/*.jpg'):
    h = hashlib.md5(open(img,'rb').read()).hexdigest()
    if h in have: continue
    lbl = img.replace('images','labels').rsplit('.',1)[0] + '.txt'
    if not os.path.exists(lbl): continue
    shutil.copy(img, 'data/C_DET/train/images/')
    shutil.copy(lbl, 'data/C_DET/train/labels/')
    n += 1
for c in glob.glob('data/C_DET/*/labels.cache'): os.remove(c)
print(f'合并 {n} 张新图, 缓存已清除')
EOF
git add -A && git commit -m "扩充训练集" && git push
```

---

## 6. 阶段四：扩充后重训

数据变多后**至少重跑两组**（论文主对比），时间充裕则 7 组全重跑（命令同阶段二，`--name` 加后缀 `_v2`，如 `G1_baseline_v2`）：

```bash
python train.py --cfg models/yolov5s.yaml      --name G1_baseline_v2 --epochs 150 --imgsz 640 --batch-size 16 --device 0
python train.py --cfg models/yolov5s_CBDG.yaml --name G4_CBDG_v2     --epochs 150 --imgsz 640 --batch-size 16 --device 0
```

> 论文写法建议：正文主表用 v2（扩充后）数据，v1（扩充前）可作为"数据增强对模型的影响"的附加讨论——两组都是真实实验，都是加分项。

---

## 7. 阶段五：结果汇总与交付

生成 `实验报告.md` 放到仓库根目录，包含：

1. **表 5.4 消融主表**（G1→G2→G3→G4，列：P / R / mAP@0.5 / mAP@0.5:0.95 / 参数量 / GFLOPs / 模型大小 / 训练时长）
2. **表 4.1 注意力对比**（G1 vs T4_SE vs G2_CA）
3. **表 4.2 BiFPN 对比**（G1 vs T4_BiFPN）
4. **表 5.3 轻量化对比**（G1 vs T5_Ghost）
5. **分类别 AP 表**（每个 run 的 crack / pinhole / spot AP，来自 val.py 输出）
6. 每组的 `results.png`、`PR_curve.png`、`confusion_matrix.png` 路径清单（论文插图直接用）
7. 训练环境记录（GPU 型号、torch 版本、训练时长）

同时把 README.md 的"结果填写"占位表替换为真实数字。最后：

```bash
git add -A && git commit -m "全部实验完成: 真实数据汇总" && git push
```

**验收：`实验报告.md` 里每个数字都能在 `runs/train/*/results.csv` 或 val 日志中找到出处。**

---

## 8. 常见故障对照

| 症状 | 原因与解决 |
|---|---|
| `CUDA error: no kernel image is available` / sm_120 报错 | torch 版本不对，5060 必须装 `--index-url https://download.pytorch.org/whl/cu128` 的 torch≥2.7 |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics==8.3.0`（本仓库 utils 硬依赖它） |
| Windows 下 dataloader 卡死/报多进程错 | 命令加 `--workers 0` 或 `--workers 2` |
| 训练中断 | `python train.py --resume runs/train/<name>/weights/last.pt` |
| numpy 2.x 相关报错 | `pip install "numpy<2"` |
| Ghost/CBDG 组前期 mAP 很低 | 正常——预训练权重大部分层不匹配，迁移率低，10~20 epoch 后会起来 |
| val.py 报 labels.cache 相关错误 | 删 `data/C_DET/*/labels.cache` 重跑 |
| 显存不足 OOM | `--batch-size 8`（并记录在报告里，所有组保持一致才可比） |

## 9. 论文章节 ↔ 实验 run 对应表

| 论文位置 | 内容 | 用哪个 run |
|---|---|---|
| 表4.1 | SE vs CA 注意力对比 | G1_baseline, T4_SE, G2_CA |
| 表4.2 | BiFPN 对比 | G1_baseline, T4_BiFPN |
| 表5.3 | GhostNet 轻量化对比 | G1_baseline, T5_Ghost |
| 表5.4 | 消融主表（4 组递进） | G1_baseline → G2_CA → G3_CA_BFPN → G4_CBDG |
| 图5.8/5.9 | 改进前后各类别 AP 对比 | G1 vs G4 的 val.py 分类别输出 |
| 图5.10/5.11 | 检测可视化 | `python detect.py --weights runs/train/G4_CBDG/weights/best.pt --source data/C_DET/valid/images --conf 0.25`，输出在 runs/detect/ |
| 表5.5 | 与 SSD/Faster R-CNN 等对比 | 本仓库不做（需要 mmdetection），论文里如需保留需另行单独实验，或改为引用原论文数据并注明 |

---

*本手册由项目整理方生成于 2026-11，代码与数据清洗状态见 git log。执行中有任何本手册未覆盖的决策，遵循原则：保持可比、如实记录、及时 commit。*
