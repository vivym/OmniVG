from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import PIL.Image
from diffusers.utils import BaseOutput


@dataclass
class VideoCrafterPipelineOutput(BaseOutput):
    """
    Output class for VideoCrafter pipelines.

    Args:
        images (`List[List[PIL.Image.Image]]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, num_frames,
            height, width, num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    frames: Union[List[List[PIL.Image.Image]], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]
