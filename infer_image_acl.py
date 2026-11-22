#!/usr/bin/env python3
"""Run a YOLOv8 OM model on one image with the CANN AscendCL Python wrapper."""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from ascend_utils.common import acl_inference


CLASS_NAMES = [
    "plastic_bottle",
    "paper_waste",
    "plastic_wrapper",
    "plastic_bag",
    "cable",
    "book",
    "stationery",
]

INPUT_SIZE = 640


def letterbox(image, size=INPUT_SIZE, color=(114, 114, 114)):
    """Resize while preserving aspect ratio and pad to a square."""
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_width = size - new_width
    pad_height = size - new_height
    left = pad_width // 2
    right = pad_width - left
    top = pad_height // 2
    bottom = pad_height - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return padded, scale, left, top


def preprocess(image):
    padded, scale, pad_left, pad_top = letterbox(image)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    tensor = np.ascontiguousarray(tensor, dtype=np.float32)
    return tensor, scale, pad_left, pad_top


def box_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_one = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(area_one + areas - intersection, 1e-7)


def class_aware_nms(boxes, scores, class_ids, iou_threshold):
    """Perform NMS independently for each class."""
    keep = []
    for class_id in np.unique(class_ids):
        indices = np.where(class_ids == class_id)[0]
        order = indices[np.argsort(scores[indices])[::-1]]
        while order.size:
            best = order[0]
            keep.append(best)
            if order.size == 1:
                break
            remaining = order[1:]
            ious = box_iou_one_to_many(boxes[best], boxes[remaining])
            order = remaining[ious <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def postprocess(raw_output, original_shape, scale, pad_left, pad_top, conf, iou):
    output = np.asarray(raw_output)
    if output.shape != (1, 11, 8400):
        raise ValueError(f"Unexpected model output shape: {output.shape}")

    predictions = output[0].T  # [8400, 11]
    xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= conf
    xywh = xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if not scores.size:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes = np.empty_like(xywh, dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale

    image_height, image_width = original_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height - 1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes = boxes[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    keep = class_aware_nms(boxes, scores, class_ids, iou)
    keep = keep[np.argsort(scores[keep])[::-1]]
    return boxes[keep], scores[keep], class_ids[keep]


def draw_detections(image, boxes, scores, class_ids):
    colors = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 200, 255),
        (255, 0, 255),
        (255, 255, 0),
        (0, 128, 255),
        (180, 0, 255),
    ]
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        color = colors[int(class_id) % len(colors)]
        label = f"{CLASS_NAMES[int(class_id)]} {float(score):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(y1, text_height + baseline + 4)
        cv2.rectangle(
            image,
            (x1, label_top - text_height - baseline - 4),
            (x1 + text_width + 6, label_top),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, label_top - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def parse_args():
    project = "/home/HwHiAiUser/python_projects/yolo_project"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=f"{project}/models/om/dianshe_yolov8n_bs1_v3.om",
    )
    parser.add_argument("--image", default=f"{project}/images/test.jpg")
    parser.add_argument("--output", default=f"{project}/outputs/test_result.jpg")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model) or os.path.getsize(args.model) == 0:
        raise FileNotFoundError(f"Valid OM model not found: {args.model}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Test image not found: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"OpenCV cannot decode image: {args.image}")
    tensor, scale, pad_left, pad_top = preprocess(image)
    print(f"Input image: {image.shape}, tensor: {tensor.shape} {tensor.dtype}")

    model = None
    acl_started = False
    try:
        acl_inference.init_acl(device_id=args.device)
        acl_started = True
        model = acl_inference.AclInference(args.model, device_id=args.device)
        print("Model inputs:", model.inputs)
        print("Model outputs:", model.outputs)

        start = time.perf_counter()
        outputs = model(tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        print("Output shapes:", [np.asarray(item).shape for item in outputs])
        print(f"NPU inference time: {elapsed_ms:.2f} ms")

        boxes, scores, class_ids = postprocess(
            outputs[0], image.shape, scale, pad_left, pad_top, args.conf, args.iou
        )
        print(f"Detections after NMS: {len(scores)}")
        for score, class_id in zip(scores, class_ids):
            print(f"  {CLASS_NAMES[int(class_id)]}: {float(score):.4f}")

        result = draw_detections(image.copy(), boxes, scores, class_ids)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if not cv2.imwrite(args.output, result):
            raise RuntimeError(f"Failed to save result: {args.output}")
        print(f"Saved result: {args.output}")
    finally:
        # The Huawei wrapper owns model/device buffers. Its documented example
        # finalizes ACL after inference; process exit releases the model object.
        if acl_started:
            acl_inference.release_acl(device_id=args.device)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
