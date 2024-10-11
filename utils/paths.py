import os


def make_eval_paths(
    video_path: str,
    database: str,
    tracking_method: str,
    sc: bool,
    sm: bool,
):
    vid_name = video_path.split("/")[-1].split(".")[0]
    print(f"Video: {vid_name}")

    save_name = f"{vid_name}.txt"
    if not sc and not sm:
        if tracking_method == "botfacesort":
            eval_save_folder = (
                f"inference/{database}/results/{database}-all/{tracking_method}_base"
            )
            fps_save_path = f"inference/{database}/results/{database}-all/{tracking_method}_base/fps.txt"
        else:
            eval_save_folder = (
                f"inference/{database}/results/{database}-all/{tracking_method}"
            )
            fps_save_path = (
                f"inference/{database}/results/{database}-all/{tracking_method}/fps.txt"
            )
    elif sc and sm:
        eval_save_folder = (
            f"inference/{database}/results/{database}-all/{tracking_method}"
        )
        fps_save_path = (
            f"inference/{database}/results/{database}-all/{tracking_method}/fps.txt"
        )
    elif sc:
        if sm:
            eval_save_folder = (
                f"inference/{database}/results/{database}-all/{tracking_method}_sc_sm"
            )
            fps_save_path = f"inference/{database}/results/{database}-all/{tracking_method}_sc_sm/fps.txt"
        else:
            eval_save_folder = (
                f"inference/{database}/results/{database}-all/{tracking_method}_sc"
            )
            fps_save_path = f"inference/{database}/results/{database}-all/{tracking_method}_sc/fps.txt"
    elif sm:
        eval_save_folder = (
            f"inference/{database}/results/{database}-all/{tracking_method}_sm"
        )
        fps_save_path = (
            f"inference/{database}/results/{database}-all/{tracking_method}_sm/fps.txt"
        )

    print(f"Save path: {eval_save_folder}")
    os.makedirs(eval_save_folder, exist_ok=True)
    eval_save_path = os.path.join(eval_save_folder, save_name)
    if os.path.exists(eval_save_path):
        os.remove(eval_save_path)

    return eval_save_path, fps_save_path
