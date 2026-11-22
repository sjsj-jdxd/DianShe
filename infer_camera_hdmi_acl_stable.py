#!/usr/bin/env python3
"""Stable real-time YOLOv8 OM inference for Orange Pi HDMI.

Features:
  - paper_waste has its own lower confidence threshold;
  - camera reader keeps only the newest frame and reconnects after failures;
  - one ACL model object is reused for all frames;
  - per-frame exceptions are isolated;
  - periodic FPS, memory, temperature and reconnect diagnostics;
  - Ctrl+C, q or Esc performs orderly cleanup.
"""

import gc
import os
import sys
import threading
import time

import cv2
import numpy as np

from ascend_utils.common import acl_inference
from infer_image_acl import CLASS_NAMES, class_aware_nms, draw_detections, preprocess


WINDOW_NAME = "Orange Pi YOLO Stable - Q or ESC to quit"
MODEL_DEFAULT = "/home/HwHiAiUser/python_projects/yolo_project/models/om/dianshe_yolov8n_bs1_v3.om"
CAMERA_DEVICE = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
NPU_DEVICE = 0
IOU_THRESHOLD = 0.40

# 固定写在代码里的分类别阈值：纸巾对应训练类别 paper_waste。
DEFAULT_CONFIDENCE = 0.50
PAPER_WASTE_CONFIDENCE = 0.30
CLASS_CONFIDENCE = {
    "paper_waste": PAPER_WASTE_CONFIDENCE,
}

CAMERA_REOPEN_DELAY = 1.0
MAX_CONSECUTIVE_READ_FAILURES = 15
STATS_INTERVAL_SECONDS = 10.0


def class_threshold(class_id):
    name = CLASS_NAMES[int(class_id)]
    return CLASS_CONFIDENCE.get(name, DEFAULT_CONFIDENCE)


def postprocess_per_class(raw_output, original_shape, scale, pad_left, pad_top, iou):
    """Decode YOLO output and apply a different confidence threshold per class."""
    output = np.asarray(raw_output)
    if output.shape != (1, 11, 8400):
        raise ValueError(f"Unexpected model output shape: {output.shape}")

    predictions = output[0].T
    xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    thresholds = np.asarray([class_threshold(cid) for cid in class_ids], dtype=np.float32)
    mask = scores >= thresholds
    xywh, scores, class_ids = xywh[mask], scores[mask], class_ids[mask]

    if not scores.size:
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
                np.empty((0,), np.int64))

    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

    height, width = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
    if not scores.size:
        return (np.empty((0, 4), np.float32), np.empty((0,), np.float32),
                np.empty((0,), np.int64))
    keep = class_aware_nms(boxes, scores, class_ids, iou)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes[keep], scores[keep], class_ids[keep]


class LatestFrameCamera:
    """Continuously capture, overwrite old frames, and reopen after failures."""

    def __init__(self, device, width, height, fps):
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.capture = None
        self.frame = None
        self.sequence = 0
        self.reconnects = 0
        self.read_failures = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="camera-reader", daemon=True)

    def _open(self):
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        failures = 0
        while self.running:
            if self.capture is None:
                self.capture = self._open()
                if self.capture is None:
                    time.sleep(CAMERA_REOPEN_DELAY)
                    continue
                self.reconnects += 1
                failures = 0
                print(f"Camera opened (reconnects={self.reconnects})", flush=True)

            ok, frame = self.capture.read()
            if ok and frame is not None:
                failures = 0
                with self.lock:
                    self.frame = frame  # overwrite: no growing queue
                    self.sequence += 1
                continue

            failures += 1
            self.read_failures += 1
            if failures >= MAX_CONSECUTIVE_READ_FAILURES:
                print("Camera read failed repeatedly; reopening /dev/video0", flush=True)
                self.capture.release()
                self.capture = None
                failures = 0
                time.sleep(CAMERA_REOPEN_DELAY)
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.sequence
            return self.frame.copy(), self.sequence

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        with self.lock:
            self.frame = None


def process_memory_mb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def temperature_c():
    values = []
    for root, _, files in os.walk("/sys/class/thermal"):
        if "temp" not in files:
            continue
        try:
            raw = int(open(os.path.join(root, "temp"), encoding="utf-8").read().strip())
            if 1000 <= raw <= 150000:
                values.append(raw / 1000.0)
        except (OSError, ValueError):
            continue
    return max(values) if values else None


def require_desktop():
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("DISPLAY is empty; launch from the HDMI desktop or pass DISPLAY=:0 and XAUTHORITY.")


def main():
    model_path = os.environ.get("YOLO_OM_MODEL", MODEL_DEFAULT)
    require_desktop()
    if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
        raise FileNotFoundError(f"Valid OM model not found: {model_path}")

    camera = None
    model = None
    acl_started = False
    try:
        print(f"Thresholds: default={DEFAULT_CONFIDENCE:.2f}, paper_waste={PAPER_WASTE_CONFIDENCE:.2f}", flush=True)
        acl_inference.init_acl(device_id=NPU_DEVICE)
        acl_started = True
        model = acl_inference.AclInference(model_path, device_id=NPU_DEVICE)
        camera = LatestFrameCamera(CAMERA_DEVICE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS).start()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        processed_sequence = -1
        smoothed_fps = 0.0
        last_stats = time.monotonic()
        frame_count = 0
        last_result = None

        while True:
            frame, sequence = camera.latest()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if frame is None or sequence == processed_sequence:
                time.sleep(0.001)
                continue

            processed_sequence = sequence
            try:
                tensor, scale, pad_left, pad_top = preprocess(frame)
                started = time.perf_counter()
                outputs = model(tensor)  # same ACL model/buffers are reused by wrapper
                infer_ms = (time.perf_counter() - started) * 1000.0
                instant_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
                smoothed_fps = instant_fps if not smoothed_fps else 0.85 * smoothed_fps + 0.15 * instant_fps
                boxes, scores, class_ids = postprocess_per_class(outputs[0], frame.shape, scale, pad_left, pad_top, IOU_THRESHOLD)
                result = draw_detections(frame, boxes, scores, class_ids)
                last_result = result
                frame_count += 1
            except Exception as error:
                print(f"Frame error (continuing): {type(error).__name__}: {error}", file=sys.stderr, flush=True)
                continue

            status = f"NPU {infer_ms:.1f} ms | {smoothed_fps:.1f} FPS | objects {len(scores)}"
            cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(result, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, result)

            now = time.monotonic()
            if now - last_stats >= STATS_INTERVAL_SECONDS:
                temp = temperature_c()
                temp_text = "n/a" if temp is None else f"{temp:.1f}C"
                print(f"stats: frames={frame_count} fps={smoothed_fps:.1f} rss={process_memory_mb():.1f}MB temp={temp_text} reconnects={camera.reconnects} read_failures={camera.read_failures}", flush=True)
                last_stats = now
                gc.collect()  # only Python objects; ACL buffers remain owned/reused by model

    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
        # Keep model alive until camera/display work has stopped, then finalize ACL.
        model = None
        gc.collect()
        if acl_started:
            acl_inference.release_acl(device_id=NPU_DEVICE)
        print("Resources released.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.", flush=True)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
