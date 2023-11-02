import os
from typing import Union, List

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from imageio.plugins.pyav import PyAVPlugin


def save_video(
    video: Union[List[Image.Image], torch.Tensor, np.ndarray],
    path: Union[str, os.PathLike],
    fps: int = 8,
    exits_ok: bool = False,
) -> None:
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy()
    elif isinstance(video, (list, tuple)):
        for frame in video:
            if not isinstance(frame, Image.Image):
                raise TypeError(f"Unsupported frame type: {type(frame)}")
        video = np.stack([np.array(img) for img in video])

    if not isinstance(video, np.ndarray):
        raise TypeError(f"Unsupported type: {type(video)}")

    if video.ndim != 4:
        raise ValueError(f"Expected 4-dimensional array, got {video.ndim}")

    if video.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3, got {video.shape[-1]}")

    if os.path.exists(path) and not exits_ok:
        raise FileExistsError(f"File already exists: {path}")

    with iio.imopen(path, "w", plugin="pyav") as out_file:
        out_file: PyAVPlugin
        out_file.init_video_stream("h264", fps=fps)

        for frame in video:
            out_file.write_frame(frame)
