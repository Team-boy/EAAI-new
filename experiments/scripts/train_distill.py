# -*- coding: utf-8 -*-
"""分辨率蒸馏: 1280 teacher → 640 student 特征对齐.
正确接入: monkey-patch student.model.loss, KD 在 forward 算 loss 时加入(可 backward).
student 中间特征用 hook 存; teacher(1280)同图前向取特征; student 上采样到 teacher 尺度做 MSE."""
import argparse
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT_YAML = "/home/fenghn/CvTest/data/Dataset B/data_split.yaml"
TEACHER_W = "/home/fenghn/CvTest/experiments/runs/opt1_hires1280/weights/best.pt"
FEAT_LAYERS = [4, 6, 9]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lambda_kd", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="distill_640")
    a = ap.parse_args()
    dev = f"cuda:{a.device}"

    student = YOLO("yolov8m.pt")
    teacher = YOLO(TEACHER_W).model.to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sstore, tstore = {}, {}

    def reg_hooks(model, store):
        for li in FEAT_LAYERS:
            def mk(idx):
                return lambda m, i, o: store.__setitem__(idx, o)
            model.model[li].register_forward_hook(mk(li))
    reg_hooks(teacher, tstore)

    state = {"hooked": False}

    def patch_loss(smodel):
        orig_loss = smodel.loss

        def new_loss(batch, preds=None):
            loss, items = orig_loss(batch, preds)   # 标准检测 loss(student 前向已触发 sstore)
            try:
                imgs = batch["img"].to(dev).float() / 255.0
                big = F.interpolate(imgs, size=(1280, 1280), mode="bilinear", align_corners=False)
                with torch.no_grad():
                    teacher(big)
                kd = 0.0
                for li in FEAT_LAYERS:
                    if li in sstore and li in tstore:
                        s = sstore[li]
                        t = tstore[li].to(s.dtype)
                        s_up = F.interpolate(s, size=t.shape[-2:], mode="bilinear", align_corners=False)
                        kd = kd + F.mse_loss(s_up, t)
                if isinstance(kd, torch.Tensor):
                    loss = loss + a.lambda_kd * kd
            except Exception as e:
                if not state.get("warned"):
                    print("KD skip:", str(e)[:80]); state["warned"] = True
            return loss, items
        smodel.loss = new_loss

    def on_start(trainer):
        if not state["hooked"]:
            reg_hooks(trainer.model, sstore)
            patch_loss(trainer.model)
            state["hooked"] = True
            print(">>> KD hooks + loss patch installed")

    student.add_callback("on_train_start", on_start)
    student.train(data=ROOT_YAML, epochs=a.epochs, imgsz=640, batch=16, device=a.device,
                  seed=0, project="/home/fenghn/CvTest/experiments/runs", name=a.name,
                  exist_ok=True, verbose=False, plots=False, workers=8)
    print("DISTILL DONE", a.name)


if __name__ == "__main__":
    main()
