#!/usr/bin/env python3
"""Hailo-8 YOLOv8 + USB Camera: record 5s video with bbox/class overlay."""

import cv2
import numpy as np
import time
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType
)

# COCO 80 class names
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

COLORS = np.random.default_rng(42).integers(0, 255, size=(80, 3)).tolist()

HEF_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
INPUT_SIZE = 640
CONF_THRESHOLD = 0.4
VIDEO_DURATION = 5  # seconds


def parse_nms_output(raw_output, img_w, img_h):
    """Parse Hailo NMS output: list of 80 classes, each with (N, 5) detections.
    Each detection: [y_min, x_min, y_max, x_max, score] in normalized [0,1].
    """
    detections = []
    for cls_id, cls_dets in enumerate(raw_output):
        arr = np.array(cls_dets)
        if arr.size == 0:
            continue
        for det in arr:
            score = det[4]
            if score < CONF_THRESHOLD:
                continue
            y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
            x1 = int(x_min * img_w)
            y1 = int(y_min * img_h)
            x2 = int(x_max * img_w)
            y2 = int(y_max * img_h)
            detections.append((cls_id, float(score), x1, y1, x2, y2))
    return detections


def draw_detections(frame, detections):
    for cls_id, score, x1, y1, x2, y2 in detections:
        color = COLORS[cls_id]
        label = f"{COCO_CLASSES[cls_id]} {score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    # Warm up camera
    for _ in range(30):
        cap.read()

    print("Loading HEF model...")
    hef = HEF(HEF_PATH)

    print("Configuring Hailo device...")
    params = VDevice.create_params()
    with VDevice(params) as vdevice:
        configure_params = ConfigureParams.create_from_hef(
            hef=hef, interface=HailoStreamInterface.PCIe
        )
        network_groups = vdevice.configure(hef, configure_params)
        network_group = network_groups[0]

        input_vstream_params = InputVStreamParams.make(
            network_group, format_type=FormatType.UINT8
        )
        output_vstream_params = OutputVStreamParams.make(
            network_group, format_type=FormatType.FLOAT32
        )

        input_info = hef.get_input_vstream_infos()[0]
        print(f"Model input: {input_info.name}, shape={input_info.shape}")

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(
            '/home/pw/ws/hailo_yolo_output.mp4', fourcc, 15.0, (640, 480)
        )

        snapshot_saved = False
        frame_count = 0
        start_time = time.time()

        print(f"Recording {VIDEO_DURATION}s with Hailo YOLOv8 inference...")

        with network_group.activate():
            from hailo_platform import InferVStreams
            with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
                while True:
                    elapsed = time.time() - start_time
                    if elapsed >= VIDEO_DURATION:
                        break

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Preprocess: resize to 640x640, keep as uint8
                    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
                    input_data = np.expand_dims(resized, axis=0)

                    input_name = input_info.name
                    raw_results = pipeline.infer({input_name: input_data})

                    # Get output
                    output_name = list(raw_results.keys())[0]
                    raw_output = raw_results[output_name][0]  # first batch

                    detections = parse_nms_output(raw_output, 640, 480)
                    annotated = draw_detections(frame.copy(), detections)

                    # Add FPS overlay
                    fps_text = f"Hailo YOLOv8s | {elapsed:.1f}s"
                    cv2.putText(annotated, fps_text, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    out_video.write(annotated)
                    frame_count += 1

                    # Save snapshot on first frame with detections (or after 1s)
                    if not snapshot_saved and (len(detections) > 0 or elapsed > 1.0):
                        cv2.imwrite('/home/pw/ws/hailo_yolo_snapshot.jpg', annotated)
                        snapshot_saved = True
                        print(f"  Snapshot saved ({len(detections)} detections)")

        if not snapshot_saved:
            # Save last frame as snapshot
            ret, frame = cap.read()
            if ret:
                cv2.imwrite('/home/pw/ws/hailo_yolo_snapshot.jpg', frame)

        out_video.release()
        cap.release()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Done! {frame_count} frames in {total_time:.1f}s ({avg_fps:.1f} FPS)")
        print(f"Video: /home/pw/ws/hailo_yolo_output.mp4")
        print(f"Snapshot: /home/pw/ws/hailo_yolo_snapshot.jpg")


if __name__ == "__main__":
    main()
