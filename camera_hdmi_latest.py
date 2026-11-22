#!/usr/bin/env python3
"""Orange Pi AI Pro: low-latency USB camera preview on the local HDMI desktop."""

import os
import sys
import threading
import time

import cv2


DEVICE = 0
WIDTH = 640
HEIGHT = 480
CAMERA_FPS = 30
WINDOW_NAME = "Orange Pi Camera - Q or ESC to quit"


class LatestFrameCamera:
    """Continuously capture frames and retain only the newest one."""

    def __init__(self, device: int):
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.capture.isOpened():
            raise RuntimeError(
                "无法打开 /dev/video0。请确认摄像头已连接，且没有被其他程序占用。"
            )

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                # Replace the previous image. Old frames never form a queue.
                self.frame = frame

    def read_latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()


def require_graphical_desktop():
    display = os.environ.get("DISPLAY")
    if not display:
        print("错误：当前终端没有连接到香橙派图形桌面，DISPLAY 为空。")
        print("请不要从串口或普通 SSH 终端运行本程序。")
        print("请在香橙派 HDMI 桌面中按 Ctrl+Alt+T 打开终端，再运行：")
        print("  cd /home/HwHiAiUser/python_projects")
        print("  python3 camera_hdmi_latest.py")
        sys.exit(2)


def main():
    require_graphical_desktop()
    camera = LatestFrameCamera(DEVICE).start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    display_fps = 0.0
    frames = 0
    fps_started = time.perf_counter()

    try:
        while True:
            frame = camera.read_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            frames += 1
            now = time.perf_counter()
            elapsed = now - fps_started
            if elapsed >= 1.0:
                display_fps = frames / elapsed
                frames = 0
                fps_started = now

            cv2.putText(
                frame,
                f"Display FPS: {display_fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
