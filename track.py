import sys
from pathlib import Path
import warnings
from torch.serialization import SourceChangeWarning

warnings.filterwarnings("ignore", category=SourceChangeWarning)
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import argparse
import torch
import cv2
from tqdm import tqdm
import numpy as np
from datetime import datetime

from utils.torch_utils import select_device
from utils.paths import make_eval_paths
from utils.constants import COLORS
from utils.histogram import calculate_histogram, chi_square_distance
from utils.seed_fix import fix_seed

from detector.scrfd import SCRFD
from tracker.tracker_zoo import create_tracker
from tracker.utils.ops import xyxy2tlwh


@torch.no_grad()
def main(args):
    fps = []
    device = select_device(args.device, batch_size=1)
    print("Using device: {}".format(device))

    model = SCRFD(args.face_detector, args.nms)

    if args.eval:
        eval_save_path, fps_save_path = make_eval_paths(
            args.video_path,
            args.database,
            args.tracking_method,
            args.sc,
            args.sm,
        )

    video = cv2.VideoCapture(args.video_path)
    frame_counts = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # For tracker
    tracking_config_path = f"tracker/configs/{args.tracking_method}.yaml"
    n_sources = 1
    tracker_list = []
    for _ in range(n_sources):
        tracker = create_tracker(
            args.tracking_method,
            tracking_config_path,
            args.reid_model,
            device,
            args.half,
            args.sc,
            args.sm,
        )
        tracker_list.append(tracker)

    outputs = [None] * n_sources

    # Run tracking
    curr_frames, prev_frames = [None] * n_sources, [None] * n_sources

    for frame_idx in tqdm(range(frame_counts)):
        start_time = datetime.now()

        if frame_idx == args.end_frame:
            break
        ret, frame = video.read()
        curr_frames[0] = frame.copy()
        curr_frame = frame.copy()
        args.line_thick = max(frame.shape[:2]) // 1000 + 3

        if ret:
            # Face detection
            face_bboxes, face_landmarks = model.detect(
                frame, args.conf, input_size=(640, 640)
            )

            if len(face_bboxes) > 0:
                # Calculate histogram distance
                curr_hist = calculate_histogram(curr_frames[0], args.num_bins)
                prev_hist = (
                    calculate_histogram(prev_frames[0], args.num_bins)
                    if prev_frames[0] is not None
                    else curr_hist
                )
                hist_dist = chi_square_distance(curr_hist, prev_hist)

                if hist_dist > args.shot_change_threshold:
                    shot_changed = True
                else:
                    shot_changed = False

                # Update tracker
                face_bboxes = np.insert(
                    face_bboxes, 5, 1.0, axis=1
                )  # insert label for tracker

                if face_landmarks is not None and args.tracking_method not in [
                    "bytetrack",
                    "ocsort",
                ]:
                    if args.sc:
                        outputs[0] = tracker_list[0].update(
                            face_bboxes,
                            curr_frames[0],
                            shot_changed=shot_changed,
                            landmarks=face_landmarks,
                        )
                    else:
                        outputs[0] = tracker_list[0].update(
                            face_bboxes, curr_frames[0], landmarks=face_landmarks
                        )
                else:
                    if args.sc:
                        outputs[0] = tracker_list[0].update(
                            face_bboxes, curr_frames[0], shot_changed=shot_changed
                        )
                    else:
                        outputs[0] = tracker_list[0].update(face_bboxes, curr_frames[0])

                if outputs[0] is not None:
                    for j, output in enumerate(outputs[0]):
                        color = COLORS[j % len(COLORS)]
                        bbox = output[:4]
                        id_ = int(output[4])
                        score = output[5]
                        if args.eval:
                            with open(eval_save_path, "a") as ef:
                                t, l, w, h = map(int, xyxy2tlwh(bbox))
                                ef.write(
                                    f"{int(frame_idx+1)},{id_},{t},{l},{w},{h},{score},-1,-1,-1\n"
                                )  # Follow MOT15 format

                        if args.display:
                            cv2.rectangle(
                                curr_frame,
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                color,
                                thickness=args.line_thick,
                            )

                            cv2.putText(
                                curr_frame,
                                str(id_),
                                (int(bbox[0]), int(bbox[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                args.line_thick,
                            )
            end_time = datetime.now()
            total_seconds = (end_time - start_time).total_seconds()
            fps.append(1 / total_seconds)

            prev_frames[0] = curr_frames[0]

        if args.display:
            cv2.putText(
                curr_frame,
                "Tracker: {} | Frame {} | Tracks: {}".format(
                    args.tracking_method,
                    frame_idx + 1,
                    len(outputs[0]) if outputs[0] is not None else 0,
                ),
                (5, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                thickness=2,
            )
            cv2.imshow("", curr_frame)
            cv2.waitKey(1)

    fps = np.mean(fps)
    if args.eval:
        with open(fps_save_path, "a") as f:
            f.write(f"{fps}\n")

    print(f"fps: {fps:.2f}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fix_seed(42)
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or cpu")
    parser.add_argument("-p", "--video-path", default="", help="path to video file")
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--database", type=str, default="music", help="database name for evaluation"
    )

    # Face Detection
    parser.add_argument(
        "--face-detector", type=Path, default="weights/scrfd_10g_gnkps.onnx"
    )
    parser.add_argument("--conf", type=float, default=0.1, help="confidence threshold")
    parser.add_argument("--nms", type=float, default=0.7, help="NMS threshold")

    # Tracking
    parser.add_argument(
        "--tracking-method",
        type=str,
        default="hybridsort",
        help="deepocsort, botsort, strongsort, ocsort, bytetrack, hybridsort, botfacesort",
    )
    parser.add_argument(
        "--reid-model", type=Path, default="weights/resnet101_ir_webface12m.ckpt"
    )

    # Proposed Methods
    parser.add_argument(
        "--num-bins",
        type=int,
        default=64,
        help="number of bins for histogram calculation",
    )
    parser.add_argument("--shot-change-threshold", type=float, default=0.4)
    parser.add_argument(
        "--sc",
        "--shot-change-detection",
        action="store_true",
        help="shot change detection",
    )
    parser.add_argument(
        "--sm", "--shared-memory", action="store_true", help="use shared memory"
    )

    # Evaluation
    parser.add_argument("--eval", action="store_true", help="eval tracking results")
    parser.add_argument(
        "--end-frame", type=int, default=-1, help="end frame for evaluation"
    )

    # Visulaization
    parser.add_argument(
        "--thickness", type=int, default=2, help="thickness of orientation lines"
    )
    parser.add_argument(
        "--display", action="store_true", help="display inference results"
    )

    args = parser.parse_args()

    main(args)
