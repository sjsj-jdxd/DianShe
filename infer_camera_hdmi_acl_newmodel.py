#!/usr/bin/env python3
"""新 OM 模型的实时 HDMI 推理。

这是新模型专用脚本，不修改项目原有的 infer_camera_hdmi_acl.py。
默认模型为 dianshe_yolov8n_bs2.om；实际转换输入是 batch=1。
请先 source 项目的 setup_cann_env.sh，再从 HDMI 图形会话运行。
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from ascend_utils.common import acl_inference
from infer_image_acl import CLASS_NAMES, class_aware_nms, draw_detections, preprocess


DEFAULT_MODEL = "/home/HwHiAiUser/python_projects/yolo_project/models/om/dianshe_yolov8n_bs2.om"


def postprocess(raw_output, original_shape, scale, pad_left, pad_top, conf, iou):
    """Decode YOLOv8 output [1, 4+classes, 8400] and map boxes to camera image."""
    output = np.asarray(raw_output)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2:
        raise ValueError(f"Unexpected model output shape: {np.asarray(raw_output).shape}")
    # Accept either [1, 4+C, N] or [1, N, 4+C] layouts.
    if output.shape[0] < output.shape[1] and output.shape[0] == 4 + len(CLASS_NAMES):
        predictions = output.T
    elif output.shape[1] == 4 + len(CLASS_NAMES):
        predictions = output
    else:
        raise ValueError(f"Output does not match {len(CLASS_NAMES)} classes: {output.shape}")

    xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    keep_mask = scores >= conf
    xywh, scores, class_ids = xywh[keep_mask], scores[keep_mask], class_ids[keep_mask]
    if not len(scores):
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)

    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    h, w = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    if not len(scores):
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    keep = class_aware_nms(boxes, scores, class_ids, iou)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes[keep], scores[keep], class_ids[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("YOLO_OM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY 为空；请先确认 HwHiAiUser 的 HDMI/XFCE 会话，"
            "并使用 DISPLAY=:0、XAUTHORITY=/home/HwHiAiUser/.Xauthority "
            "运行，而不是直接从串口运行。"
        )
    if not os.path.isfile(args.model) or os.path.getsize(args.model) == 0:
        raise FileNotFoundError(f"Valid OM model not found: {args.model}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 /dev/video{args.camera}")

    acl_started = False
    model = None
    window = "Orange Pi YOLO - Q/ESC退出"
    try:
        acl_inference.init_acl(device_id=args.device)
        acl_started = True
        model = acl_inference.AclInference(args.model, device_id=args.device)
        print(f"Model: {args.model}", flush=True)
        print(f"Inputs: {model.inputs}; outputs: {model.outputs}", flush=True)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frames, fps, last_time = 0, 0.0, time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            tensor, scale, pad_left, pad_top = preprocess(frame)
            start = time.perf_counter()
            outputs = model(tensor)
            infer_ms = (time.perf_counter() - start) * 1000
            boxes, scores, class_ids = postprocess(outputs[0], frame.shape, scale, pad_left, pad_top, args.conf, args.iou)
            result = draw_detections(frame, boxes, scores, class_ids)
            frames += 1
            elapsed = time.perf_counter() - last_time
            if elapsed >= 1.0:
                fps, frames, last_time = frames / elapsed, 0, time.perf_counter()
            cv2.putText(result, f"NPU {infer_ms:.1f} ms | FPS {fps:.1f} | objects {len(scores)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, result)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        model = None
        if acl_started:
            acl_inference.release_acl(device_id=args.device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
