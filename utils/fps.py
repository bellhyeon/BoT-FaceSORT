from argparse import ArgumentParser


def get_fps(txt_path, database):
    fps = 0
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            value = float(line)
            fps += value

    f.close()
    if "Movie" in database:
        fps /= 10.0
    elif "Music" in database:
        fps /= 8.0
    elif "Choke" in database:
        fps /= 6.0

    return fps


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--database", type=str)
    parser.add_argument("--tracker", type=str)
    args = parser.parse_args()

    txt_path = f"./inference/{args.database}/results/{args.database}-all/{args.tracker}/fps.txt"

    print(get_fps(txt_path, args.database))
