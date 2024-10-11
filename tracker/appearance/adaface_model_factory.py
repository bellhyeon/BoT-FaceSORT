import sys
import time
import torch

from tracker.utils import logger as LOGGER

__model_types = ["resnet18_ir", "resnet50_ir", "resnet101_ir"]

__trained_urls = {
    # resnet18_ir
    "resnet18_ir_casia.ckpt": "https://drive.google.com/uc?id=1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe",
    "resnet18_ir_vgg2.ckpt": "https://drive.google.com/uc?id=1k7onoJusC0xjqfjB-hNNaxz9u6eEzFdv",
    "resnet18_ir_webface4m.ckpt": "https://drive.google.com/uc?id=1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y",
    # resnet50_ir
    "resnet50_ir_casia.ckpt": "https://drive.google.com/uc?id=1g1qdg7_HSzkue7_VrW64fnWuHl0YL2C2",
    "resnet50_ir_webface4m.ckpt": "https://drive.google.com/uc?id=1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN",
    "resnet50_ir_ms1mv2.ckpt": "https://drive.google.com/uc?id=1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI",
    # resnet101_ir
    "resnet101_ir_ms1mv2.ckpt": "https://drive.google.com/uc?id=1m757p4-tUU5xlSHLaO04sqnhvqankimN",
    "resnet101_ir_ms1mv3.ckpt": "https://drive.google.com/uc?id=1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI",
    "resnet101_ir_webface4m.ckpt": "https://drive.google.com/uc?id=18jQkqB0avFqWa0Pas52g54xNshUOQJpQ",
    "resnet101_ir_webface12m.ckpt": "https://drive.google.com/uc?id=1dswnavflETcnAuplZj1IOKKP0eM8ITgT",
}


def show_downloadable_models():
    LOGGER.info("\nAvailable .pt Face ReID models for automatic download")
    LOGGER.info(list(__trained_urls.keys()))


def get_model_url(model):
    if model.name in __trained_urls:
        return __trained_urls[model.name]
    else:
        None


def is_model_in_model_types(model):
    if model.name in __model_types:
        return True
    else:
        return False


def get_model_name(model):
    for x in __model_types:
        if x in model.name:
            return x
    return None


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    LOGGER.info('* url="{}"'.format(url))
    LOGGER.info('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write("\n")


def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from boxmot.appearance.backbones import build_model
        >>> from boxmot.appearance.reid_model_factory import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> model = build_model()
        >>> load_pretrained_weights(model, weight_path)
    """

    if not torch.cuda.is_available():
        checkpoint = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        checkpoint = torch.load(weight_path)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = {
        key[6:]: val for key, val in state_dict.items() if key.startswith("model.")
    }

    model.load_state_dict(model_dict)
    LOGGER.success(f'Successfully loaded pretrained weights from "{weight_path}"')
