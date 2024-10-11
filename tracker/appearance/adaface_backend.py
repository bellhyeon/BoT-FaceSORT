from os.path import exists as file_exists
from pathlib import Path

import cv2
import gdown
import numpy as np
import torch
import torch.nn as nn
import os
import onnxruntime

from .adaface_model_factory import (
    get_model_name,
    get_model_url,
    load_pretrained_weights,
    show_downloadable_models,
)
from .adaface import build_model
from tracker.utils import logger as LOGGER
from .utils.align_trans import warp_and_crop_face, get_reference_facial_points


def check_suffix(file="resnet50_ir_webface4m.ckpt", suffix=(".pt",), msg=""):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                try:
                    assert s in suffix
                except AssertionError as err:
                    LOGGER.error(f"{err}{f} acceptable suffix is {suffix}")


class AdafaceBackendTorch(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights="resnet50_ir_webface4m.ckpt",
        device=torch.device("cpu"),
        fp16=False,
    ):
        super().__init__()

        w = weights[0] if isinstance(weights, list) else weights
        self.ckpt = self.model_type(w)  # get backend

        self.fp16 = fp16
        self.device = device
        model_name = get_model_name(w)

        if w.suffix == ".ckpt":
            model_url = get_model_url(w)
            if not file_exists(w) and model_url is not None:
                gdown.download(model_url, str(w), quiet=False)
            elif file_exists(w):
                pass
            else:
                LOGGER.error(
                    f"No URL associated to the chosen Face ReID weights ({w}). Choose between:"
                )
                show_downloadable_models()
                exit()

        # Build model
        self.model = build_model(model_name)

        if self.ckpt:  # PyTorch
            # populate model arch with weights
            if w and w.is_file() and w.suffix == ".ckpt":
                load_pretrained_weights(self.model, w)
                self.model.to(device).eval()
                self.model.half() if self.fp16 else self.model.float()
        else:
            LOGGER.error("This model framework is not supported yet!")
            exit()

    @staticmethod
    def model_type(p="path/to/model.ckpt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        from . import export_formats

        sf = list(export_formats().Suffix)  # export suffixes
        check_suffix(p, sf)  # checks
        types = [s in Path(p).name for s in sf]
        return types

    def preprocess(self, xyxys, img):
        crops = []
        h, w = img.shape[:2]
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]
            # resize
            try:
                crop = cv2.resize(
                    crop,
                    (112, 112),  # from (x, y) to (128, 256) | (w, h)
                    interpolation=cv2.INTER_LINEAR,
                )
            except cv2.error:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)

            crop = crop[:, :, ::-1]  # RGB to BGR for input
            # normalization & standardization
            crop = ((crop / 255.0) - 0.5) / 0.5

            crop = torch.from_numpy(crop).float()
            crops.append(crop)

        crops = torch.stack(crops, dim=0)
        crops = torch.permute(crops, (0, 3, 1, 2))
        crops = crops.to(
            dtype=torch.half if self.fp16 else torch.float, device=self.device
        )

        return crops

    def forward(self, im_batch):
        # batch to half
        if self.fp16 and im_batch.dtype != torch.float16:
            im_batch = im_batch.half()

        # batch processing
        features = []
        if self.ckpt:
            features = self.model(im_batch)
        else:
            LOGGER.error(
                "Framework not supported at the moment, leave an enhancement suggestion"
            )
            exit()

        if isinstance(features, (list, tuple)):
            return (
                self.to_numpy(features[0])
                if len(features) == 1
                else [self.to_numpy(x) for x in features]
            )
        else:
            return self.to_numpy(features)

    def to_numpy(self, x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    @torch.no_grad()
    def get_features(self, img, xyxys, landmarks=None):
        if xyxys.size != 0:
            if landmarks is not None and xyxys.shape[0] == landmarks.shape[0]:
                assert xyxys.shape[0] == landmarks.shape[0]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                reference = get_reference_facial_points(default_square=True)
                crops = []
                for landmark in landmarks:
                    # facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]

                    warped_face = warp_and_crop_face(
                        np.array(img), landmark, reference, crop_size=(112, 112)
                    )

                    warped_face = warped_face[:, :, ::-1]  # RGB to BGR for input

                    warped_face = ((warped_face / 255.0) - 0.5) / 0.5

                    warped_face = torch.from_numpy(warped_face).float()
                    crops.append(warped_face)

                crops = torch.stack(crops, dim=0)
                crops = torch.permute(crops, (0, 3, 1, 2))
                crops = crops.to(
                    dtype=torch.half if self.fp16 else torch.float, device=self.device
                )
            else:
                crops = self.preprocess(xyxys, img)
            features, _ = self.forward(crops)
        else:
            features = np.array([])
        # features = features / np.linalg.norm(features)
        return features


class AdafaceBackendONNX(nn.Module):
    # ReID models MultiBackend class for python inference on various backends
    def __init__(self, weights="resnet100_ir_webface12m.onnx", session=None):
        super().__init__()

        self.session = session
        if self.session is None:
            assert weights is not None
            assert os.path.exists(weights)
            LOGGER.success(f'Successfully loaded pretrained weights from "{weights}"')

            so = onnxruntime.SessionOptions()
            exproviders = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = onnxruntime.InferenceSession(
                weights, so, providers=exproviders
            )

    @staticmethod
    def preprocess(xyxys, img):
        crops = []
        h, w = img.shape[:2]
        # dets are of different sizes so batch preprocessing is not possible
        for box in xyxys:
            x1, y1, x2, y2 = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            crop = img[y1:y2, x1:x2]

            # resize
            try:
                crop = cv2.resize(
                    crop,
                    (112, 112),  # from (x, y) to (128, 256) | (w, h)
                    interpolation=cv2.INTER_LINEAR,
                )
            except cv2.error:
                crop = np.zeros((112, 112, 3), dtype=np.uint8)

            # normalization & standardization
            crop = ((crop / 255.0) - 0.5) / 0.5
            crop = np.asarray(crop, dtype=np.float32)
            crops.append(crop)

        crops = np.stack(crops, axis=0)

        crops = np.transpose(crops, (0, 3, 1, 2))
        return crops

    def forward(self, im_batch):
        features = self.session.run(None, {"input": im_batch})
        return np.asarray(features[0], dtype=np.float32)

    @torch.no_grad()
    def get_features(self, img, xyxys, landmarks=None):
        if xyxys.size != 0:
            if landmarks is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                reference = get_reference_facial_points(default_square=True)
                crops = []
                for landmark in landmarks:
                    # facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
                    warped_face = warp_and_crop_face(
                        np.array(img), landmark, reference, crop_size=(112, 112)
                    )
                    warped_face = warped_face[:, :, ::-1]  # RGB to BGR for input
                    warped_face = ((warped_face / 255.0) - 0.5) / 0.5

                    warped_face = np.asarray(warped_face, dtype=np.float32)
                    crops.append(warped_face)

                crops = np.stack(crops, axis=0)
                crops = np.transpose(crops, (0, 3, 1, 2))
            else:
                crops = self.preprocess(xyxys, img)
            features = self.forward(crops)
        else:
            features = np.array([])
        # features = features / np.linalg.norm(features)
        return features
