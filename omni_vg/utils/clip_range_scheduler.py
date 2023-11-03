from typing import Generator, Optional, Union

import numpy as np
import torch


# Returns fraction that has denominator that is a power of 2
def ordered_halving(val: int) -> float:
    # get binary value, padded with 0s for 64 bits
    bin_str = f"{val:064b}"
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    final = as_int / (1 << 64)
    return final


class ClipRangeScheduler:
    def __init__(
        self,
        num_steps: int,
        num_frames: int,
        clip_length: int = 16,
        clip_stride: int = 1,
        clip_overlap: int = 8,
        closed_loop: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.num_frames = num_frames
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.clip_overlap = clip_overlap
        self.closed_loop = closed_loop
        self.device = device

    def iter(self, step: int) -> Generator[torch.Tensor, None, None]:
        raise NotImplementedError


class UniformClipRangeScheduler(ClipRangeScheduler):
    def iter(self, step: int) -> Generator[torch.Tensor, None, None]:
        if self.num_frames <= self.clip_length:
            yield list(range(self.num_frames))
            return

        clip_stride = min(
            self.clip_stride, int(np.ceil(np.log2(self.num_frames / self.clip_length))) + 1
        )

        for context_step in 1 << np.arange(clip_stride):
            pad = int(round(self.num_frames * ordered_halving(step)))
            for j in range(
                int(ordered_halving(step) * context_step) + pad,
                self.num_frames + pad + (0 if self.closed_loop else -self.clip_overlap),
                (self.clip_length * context_step - self.clip_overlap),
            ):
                yield [
                    e % self.num_frames
                    for e in range(j, j + self.clip_length * context_step, context_step)
                ]


def get_context_scheduler(
    scheduler_name: str,
    num_steps: int,
    num_frames: int,
    clip_length: int = 16,
    clip_stride: int = 1,
    clip_overlap: int = 8,
    closed_loop: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> ClipRangeScheduler:
    if scheduler_name == "uniform":
        return UniformClipRangeScheduler(
            num_steps=num_steps,
            num_frames=num_frames,
            clip_length=clip_length,
            clip_stride=clip_stride,
            clip_overlap=clip_overlap,
            closed_loop=closed_loop,
            device=device,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
