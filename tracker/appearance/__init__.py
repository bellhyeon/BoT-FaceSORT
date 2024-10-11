import pandas as pd


def export_formats():
    # yolo tracking export formats
    x = [["PyTorch", "-", ".ckpt", True, True]]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])
