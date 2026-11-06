# YOLOv5 半自动标注工具: 用已训练模型为未标注图片池生成预标注
# 用法: 先在 5060 上完成消融训练, 再用最好的 best.pt 预标注数据池:
#   python autolabel.py --weights runs/train/G4_CBDG/weights/best.pt \
#       --source D:/datasets/dataset --conf 0.4 --outdir autolabel_out
# 输出:
#   autolabel_out/labels/*.txt   YOLO 检测格式预标注 (类别 xc yc w h)
#   autolabel_out/review.csv     复核优先级清单 (建议从"需重点复核"开始看)
# 人工在 labelImg/X-AnyLabeling 中加载图片+预标注, 校对后并入 data/C_DET/train/

import argparse
import csv
from pathlib import Path

import torch

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import LOGGER, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# 与 data/C_DET/data.yaml 保持一致
NAMES = {0: "crack", 1: "pinhole", 2: "spot"}


@torch.no_grad()
def run(weights, source, outdir, conf_thres=0.4, iou_thres=0.45, imgsz=640, device=""):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    names = model.names if model.names else NAMES

    outdir = Path(outdir)
    (outdir / "labels").mkdir(parents=True, exist_ok=True)
    imgs = sorted(list(Path(source).glob("*.[jp][pn]g")) + list(Path(source).glob("*.jpeg")))
    LOGGER.info(f"待预标注图片: {len(imgs)} 张, 模型: {weights}")

    rows = []
    for i, p in enumerate(imgs):
        import cv2

        im0 = cv2.imread(str(p))
        if im0 is None:
            LOGGER.warning(f"读取失败, 跳过: {p}")
            continue
        im = letterbox(im0, imgsz, stride=int(model.stride))[0]
        im = im.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        im = torch.from_numpy(im.copy()).float().div(255).unsqueeze(0).to(device)

        pred = model(im, augment=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        lines, min_conf, cls_cnt = [], 1.0, {}
        for det in pred:  # 每图一个 batch
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det.tolist():
                    x1, y1, x2, y2 = xyxy
                    h, w = im0.shape[:2]
                    lines.append(
                        f"{int(cls)} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                        f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
                    )
                    min_conf = min(min_conf, conf)
                    cls_cnt[names.get(int(cls), int(cls))] = cls_cnt.get(names.get(int(cls), int(cls)), 0) + 1
        (outdir / "labels" / (p.stem + ".txt")).write_text("\n".join(lines))

        # 复核优先级: 无检出(可能漏标/背景)和低置信度的排前面
        priority = 0 if not lines else (1 if min_conf < 0.6 else 2)
        tip = {0: "无检出: 人工确认是否漏标或背景图", 1: "低置信: 重点核对框和类别", 2: "高置信: 快速过一遍即可"}
        rows.append([p.name, len(lines), f"{min_conf:.2f}" if lines else "-", ",".join(f"{k}x{v}" for k, v in cls_cnt.items()) or "-", priority, tip[priority]])

        if (i + 1) % 200 == 0:
            LOGGER.info(f"  {i+1}/{len(imgs)}")

    rows.sort(key=lambda r: r[4])  # 按优先级升序
    with open(outdir / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w_ = csv.writer(f)
        w_.writerow(["图片", "检出框数", "最低置信度", "类别分布", "复核优先级(0最需人工)", "建议"])
        w_.writerows(rows)
    LOGGER.info(f"完成: 预标注 {outdir/'labels'}, 复核清单 {outdir/'review.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="训练好的 best.pt")
    ap.add_argument("--source", required=True, help="未标注图片目录")
    ap.add_argument("--outdir", default="autolabel_out")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    opt = ap.parse_args()
    run(opt.weights, opt.source, opt.outdir, opt.conf, imgsz=opt.imgsz, device=opt.device)
