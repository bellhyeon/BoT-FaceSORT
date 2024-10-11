import argparse
import os
from xml.etree.ElementTree import parse
from glob import glob
import cv2


def xml_to_mot(args):
    xml_file_paths = glob(os.path.join(args.gt_folder, "*.xml"))
    for xml_file_path in xml_file_paths:
        if "bbt" in xml_file_path:
            continue
        tree = parse(xml_file_path)
        root = tree.getroot()
        vid_name = xml_file_path.split("/")[-1].split(".")[0].split("_")[0]
        gt_save_folder = os.path.join(args.data_folder, vid_name, "gt")
        gt_save_path = os.path.join(gt_save_folder, "gt.txt")
        print(f"gt_save_path: {gt_save_path}, gt_save_folder: {gt_save_folder}")
        os.makedirs(gt_save_folder, exist_ok=True)
        data = []
        trajectories = root.findall("Trajectory")
        for trajectory in trajectories:
            frames = trajectory.findall("Frame")
            for frame in frames:
                frame_no = int(frame.get("frame_no"))
                obj_id = trajectory.get("obj_id")
                l = int(frame.get("x"))
                t = int(frame.get("y"))
                w = int(frame.get("width"))
                h = int(frame.get("height"))
                data.append((frame_no, obj_id, l, t, w, h))
        data.sort(key=lambda x: x[0])

        with open(gt_save_path, "w") as f:
            for d in data:
                f.write(f"{d[0]},{d[1]},{d[2]},{d[3]},{d[4]},{d[5]},1,-1,-1,-1\n")


def generate_seqinfo(args):
    vid_paths = glob(os.path.join(args.video_folder, "*"))

    for v_path in vid_paths:
        vid_name = v_path.split("/")[-1].split(".")[0]
        seq_save_folder = os.path.join(args.data_folder, vid_name)
        seq_save_path = os.path.join(seq_save_folder, "seqinfo.ini")
        xml_file_path = os.path.join(args.gt_folder, f"{vid_name}_gt.xml")
        tree = parse(xml_file_path)
        root = tree.getroot()
        seqlen = int(root.get("end_frame"))
        video = cv2.VideoCapture(v_path)
        fps = int(video.get(cv2.CAP_PROP_FPS))
        w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        with open(seq_save_path, "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name={vid_name}\n")
            f.write("imDir=img1\n")
            f.write(f"frameRate={fps}\n")
            f.write(f"seqLength={seqlen}\n")
            f.write(f"imWidth={w}\n")
            f.write(f"imHeight={h}\n")
            f.write("imExt=.jpg\n")
        f.close()
        # with open("")

    # xml_file_paths = glob(os.path.join(gt_path, "*.xml"))
    # for xml_file_path in xml_file_paths:
    #     tree = parse(xml_file_path)
    #     root = tree.getroot()
    #     video_name = xml_file_path.split("/")[-1].split(".")[0].split("_")[0]
    #     root_path = xml_file_path.split("/gt")[0]
    #     gt_save_folder = os.path.join(root_path, video_name, "gt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-folder", type=str, default="test_data/Music/gt")
    parser.add_argument("--video-folder", type=str, default="test_data/Music/videos")
    parser.add_argument("--data-folder", type=str, default="test_data/Music/Music-all")
    args = parser.parse_args()
    print("*** Converting XML to MOT15 format for Music dataset ***")
    xml_to_mot(args)
    print("*** Generating seqinfo for Music dataset ***")
    generate_seqinfo(args)
    print("*** Done ***")
