import cv2
import argparse
import os
from glob import glob


def merge_imgs(src_dir, out_path):
    frames = []
    paths = sorted(
        glob(f"./{src_dir}/*.jpg"), key=lambda x: int(x.split("/")[-1].split(".")[0])
    )
    if not os.path.exists("/".join(out_path.split("/")[:-1])):
        os.makedirs("/".join(out_path.split("/")[:-1]))
    for path in paths:
        img = cv2.imread(path)
        height, width, _ = img.shape
        size = (width, height)
        frames.append(img)

    vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for frame in frames:
        vid_writer.write(frame)
    vid_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)

    args = parser.parse_args()
    print("*** Merging images to video for ChokePoint Dataset ***")
    merge_imgs(args.src_dir, args.out_path)
    print("*** Done ***")