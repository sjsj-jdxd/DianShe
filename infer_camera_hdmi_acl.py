#!/usr/bin/env python3
"""Low-latency YOLOv8 OM inference from /dev/video0 to the local HDMI desktop."""

import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from ascend_utils.common import acl_inference
from infer_image_acl import draw_detections, postprocess, preprocess


WINDOW_NAME = "Orange Pi YOLO - Q or ESC to quit"


class LatestFrameCamera:
    """Capture continuously and overwrite the previous frame to avoid latency buildup."""

    def __init__(self, device, width, height, fps):
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {device}. Check /dev/video0 and other camera processes."
            )

        self.frame = None
        self.sequence = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = frame
                self.sequence += 1

    def latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.sequence
            return self.frame.copy(), self.sequence

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()


def require_desktop():
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "DISPLAY is empty. Run this program in the Orange Pi HDMI desktop "
            "or launch it through SSH with the desktop DISPLAY/XAUTHORITY/DBUS environment."
        )


def parse_args():
    project = "/home/HwHiAiUser/python_projects/yolo_project"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=f"{project}/models/om/dianshe_yolov8n_bs1_v3.om",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--npu-device", type=int, default=0)
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Show the result full-screen on the HDMI display.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    require_desktop()
    if not os.path.isfile(args.model) or os.path.getsize(args.model) == 0:
        raise FileNotFoundError(f"Valid OM model not found: {args.model}")

    camera = None
    model = None
    acl_started = False
    try:
        acl_inference.init_acl(device_id=args.npu_device)
        acl_started = True
        model = acl_inference.AclInference(args.model, device_id=args.npu_device)
        print("Model inputs:", model.inputs)
        print("Model outputs:", model.outputs)

        camera = LatestFrameCamera(
            args.device, args.width, args.height, args.camera_fps
        ).start()
        print(
            "Camera requested:",
            f"{args.width}x{args.height}@{args.camera_fps}",
        )

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

        processed_sequence = -1
        smoothed_fps = 0.0
        while True:
            frame, sequence = camera.latest()
            if frame is None or sequence == processed_sequence:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                time.sleep(0.001)
                continue

            # Always infer the newest available frame; intermediate frames are discarded.
            processed_sequence = sequence
            tensor, scale, pad_left, pad_top = preprocess(frame)

            started = time.perf_counter()
            outputs = model(tensor)
            infer_ms = (time.perf_counter() - started) * 1000.0
            instant_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0
            smoothed_fps = (
                instant_fps
                if smoothed_fps == 0.0
                else 0.85 * smoothed_fps + 0.15 * instant_fps
            )

            boxes, scores, class_ids = postprocess(
                outputs[0],
                frame.shape,
                scale,
                pad_left,
                pad_top,
                args.conf,
                args.iou,
            )
            result = draw_detections(frame, boxes, scores, class_ids)

            status = (
                f"NPU {infer_ms:.1f} ms | {smoothed_fps:.1f} FPS | "
                f"objects {len(scores)}"
            )
            cv2.rectangle(result, (0, 0), (result.shape[1], 38), (0, 0, 0), -1)
            cv2.putText(
                result,
                status,
                (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, result)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
        # The Huawei wrapper owns model buffers. Keep the object alive until all
        # inference and display work is finished, then finalize ACL at process end.
        if acl_started:
            acl_inference.release_acl(device_id=args.npu_device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
