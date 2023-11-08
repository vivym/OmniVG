import fnmatch
import os
import re
import importlib
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from accelerate import init_empty_weights
from diffusers.pipelines.pipeline_utils import (
    DiffusionPipeline,
    is_safetensors_compatible,
    variant_compatible_siblings,
)
from diffusers.utils import HF_HUB_OFFLINE, is_accelerate_available, is_torch_version, logging
from diffusers.utils.constants import DIFFUSERS_CACHE
from huggingface_hub import hf_hub_download, model_info, snapshot_download
from requests.exceptions import HTTPError
from safetensors import safe_open

logger = logging.get_logger(__name__)


class AnimateDiffBasePipeline(DiffusionPipeline):
    pipeline_config_name: str = "pipeline.json"

    @classmethod
    def from_pretrained(
        cls,
        base_pipeline: Union[str, os.PathLike, DiffusionPipeline],
        motion_module: Optional[Union[str, os.PathLike]],
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        custom_pipeline: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Union[str, os.PathLike] = DIFFUSERS_CACHE,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        output_loading_info: bool = False,
        local_files_only: bool = HF_HUB_OFFLINE,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        custom_revision: str = "main",
        from_flax: bool = False,
        device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]] = None,
        max_memory: Optional[dict] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = None,
        offload_state_dict: bool = True,
        low_cpu_mem_usage: bool = True,
        use_safetensors: Optional[bool] = None,
        use_onnx: Optional[bool] = None,
        variant: Optional[str] = None,
        **kwargs,
    ) -> "AnimateDiffBasePipeline":
        # 1. Prepare base pipeline
        if not isinstance(base_pipeline, DiffusionPipeline):
            base_pipeline: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path=base_pipeline,
                torch_dtype=torch_dtype,
                custom_pipeline=custom_pipeline,
                force_download=force_download,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                custom_revision=custom_revision,
                from_flax=from_flax,
                device_map=device_map,
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=offload_state_dict,
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                variant=variant,
                **kwargs,
            )

        # 2. Download the motion_module and configs
        if not os.path.isdir(motion_module):
            if motion_module.count("/") > 1:
                raise ValueError(
                    f'The provided motion_module "{motion_module}"'
                    " is neither a valid local path nor a valid repo id. Please check the parameter."
                )

            cached_folder = cls.download_motion_module(
                motion_module,
                cache_dir=cache_dir,
                resume_download=resume_download,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                from_flax=from_flax,
                use_safetensors=use_safetensors,
                use_onnx=use_onnx,
                custom_pipeline=custom_pipeline,
                custom_revision=custom_revision,
                variant=variant,
                load_connected_pipeline=False,
                **kwargs,
            )
        else:
            cached_folder = motion_module

        pipeline_config_dict = cls._dict_from_json_file(os.path.join(cached_folder, cls.pipeline_config_name))

        # 3. Define which model components should load variants
        # We retrieve the information by matching whether variant
        # model checkpoints exist in the subfolders
        model_variants = {}
        if variant is not None:
            for folder in os.listdir(cached_folder):
                folder_path = os.path.join(cached_folder, folder)
                is_folder = os.path.isdir(folder_path) and folder in ["mm"]
                variant_exists = is_folder and any(
                    p.split(".")[1].startswith(variant) for p in os.listdir(folder_path)
                )
                if variant_exists:
                    model_variants[folder] = variant

        # 4. Load the pipeline class
        pipeline_cls = cls.get_pipeline_or_model_cls(pipeline_config_dict["_class_name"])

        expected_modules, _ = cls._get_signature_keys(pipeline_cls)

        # 5. Prepare init_dict
        init_dict = {}
        for module_name in expected_modules:
            if module_name == "unet":
                continue

            if module_name in kwargs:
                module = kwargs[module_name]
            else:
                module = getattr(base_pipeline, module_name)
            init_dict[module_name] = module

        if "scheduler" in pipeline_config_dict:
            library_name, class_name = pipeline_config_dict["scheduler"]
            scheduler_config_path = os.path.join(cached_folder, "scheduler", "scheduler_config.json")
            scheduler_config = cls._dict_from_json_file(scheduler_config_path)
            assert scheduler_config["_class_name"] == class_name

            scheduler_cls = cls.get_pipeline_or_model_cls(class_name, library_name=library_name)
            init_dict["scheduler"] = scheduler_cls(**scheduler_config)

        # 6. Throw nice warnings / errors for fast accelerate loading
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        if device_map is not None and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `device_map=None`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is False and device_map is not None:
            raise ValueError(
                f"You cannot set `low_cpu_mem_usage` to False while using device_map={device_map} for loading and"
                " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
            )

        # 7. Load motion module and configs
        config_dict = cls._dict_from_json_file(os.path.join(cached_folder, "mm", "config.json"))

        base_unet = kwargs["unet"] if "unet" in kwargs else base_pipeline.unet
        unet_init_dict = {
            key: value
            for key, value in base_unet.config.items()
            if not key.startswith("_")
        }
        unet_init_dict.update({
            key: value
            for key, value in config_dict.items()
            if not key.startswith("_")
        })

        for key in ["down_block_types", "mid_block_type", "up_block_types"]:
            if key in unet_init_dict:
                if isinstance(unet_init_dict[key], str):
                    unet_init_dict[key] = unet_init_dict[key].replace("2D", "3D")
                else:
                    unet_init_dict[key] = [
                        cls_name.replace("2D", "3D")
                        for cls_name in unet_init_dict[key]
                    ]

        unet_cls = cls.get_pipeline_or_model_cls(config_dict["_class_name"])

        assert "attention_head_dim" not in unet_init_dict

        with init_empty_weights():
            unet = unet_cls(**unet_init_dict)

        state_dict = base_unet.state_dict()

        # TODO: support variants
        with safe_open(os.path.join(cached_folder, "mm", "diffusion_pytorch_model.safetensors"), framework="pt", device="cpu") as f:
            for key in logging.tqdm(list(f.keys()), desc="Loading motion module..."):
                state_dict[key] = f.get_tensor(key)

        unet.load_state_dict(state_dict, assign=True)

        init_dict["unet"] = unet

        # 8. Instantiate the pipeline
        return pipeline_cls(**init_dict)

    @classmethod
    def download_motion_module(
        cls,
        motion_module: str,
        cache_dir: Union[str, os.PathLike] = DIFFUSERS_CACHE,
        resume_download: bool = False,
        force_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        local_files_only: bool = HF_HUB_OFFLINE,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        from_flax: bool = False,
        use_safetensors: Optional[bool] = None,
        use_onnx: Optional[bool] = None,
        custom_pipeline: Optional[str] = None,
        custom_revision: str = "main",
        variant: Optional[str] = None,
        **kwargs,
    ) -> Union[str, os.PathLike]:
        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        allow_patterns = None
        ignore_patterns = None

        model_info_call_error: Optional[Exception] = None
        if not local_files_only:
            try:
                info = model_info(
                    motion_module,
                    use_auth_token=use_auth_token,
                    revision=revision,
                )
            except HTTPError as e:
                logger.warn(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
                local_files_only = True
                model_info_call_error = e  # save error to reraise it if model is not cached locally

        filenames = {sibling.rfilename for sibling in info.siblings}
        model_filenames, variant_filenames = variant_compatible_siblings(filenames, variant=variant)

        if not local_files_only:
            pipeline_config_file = hf_hub_download(
                motion_module,
                cls.pipeline_config_name,
                cache_dir=cache_dir,
                revision=revision,
                proxies=proxies,
                force_download=force_download,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
            )

            pipeline_config_dict = cls._dict_from_json_file(pipeline_config_file)

            pipeline_cls_name = pipeline_config_dict["_class_name"]
            pipeline_cls_name = pipeline_cls_name[4:] if pipeline_cls_name.startswith("Flax") else pipeline_cls_name

            pipeline_cls = cls.get_pipeline_or_model_cls(pipeline_cls_name)
            expected_components, _ = cls._get_signature_keys(pipeline_cls)
            passed_components = [k for k in expected_components if k in kwargs]

            if (
                use_safetensors
                and not allow_pickle
                and not is_safetensors_compatible(
                    model_filenames, variant=variant, passed_components=passed_components
                )
            ):
                raise EnvironmentError(
                    f"Could not found the necessary `safetensors` weights in {model_filenames} (variant={variant})"
                )

            if from_flax:
                ignore_patterns = ["*.bin", "*.safetensors", "*.onnx", "*.pb"]
            elif use_safetensors and is_safetensors_compatible(
                model_filenames, variant=variant, passed_components=passed_components
            ):
                ignore_patterns = ["*.bin", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_cls._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                safetensors_variant_filenames = {f for f in variant_filenames if f.endswith(".safetensors")}
                safetensors_model_filenames = {f for f in model_filenames if f.endswith(".safetensors")}
                if (
                    len(safetensors_variant_filenames) > 0
                    and safetensors_model_filenames != safetensors_variant_filenames
                ):
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(safetensors_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(safetensors_model_filenames - safetensors_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )
            else:
                ignore_patterns = ["*.safetensors", "*.msgpack"]

                use_onnx = use_onnx if use_onnx is not None else pipeline_cls._is_onnx
                if not use_onnx:
                    ignore_patterns += ["*.onnx", "*.pb"]

                bin_variant_filenames = {f for f in variant_filenames if f.endswith(".bin")}
                bin_model_filenames = {f for f in model_filenames if f.endswith(".bin")}
                if len(bin_variant_filenames) > 0 and bin_model_filenames != bin_variant_filenames:
                    logger.warn(
                        f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n[{', '.join(bin_variant_filenames)}]\nLoaded non-{variant} filenames:\n[{', '.join(bin_model_filenames - bin_variant_filenames)}\nIf this behavior is not expected, please check your folder structure."
                    )

            # Don't download any objects that are passed
            allow_patterns = [
                p for p in allow_patterns if not (len(p.split("/")) == 2 and p.split("/")[0] in passed_components)
            ]

            # Don't download index files of forbidden patterns either
            ignore_patterns = ignore_patterns + [f"{i}.index.*json" for i in ignore_patterns]

            re_ignore_pattern = [re.compile(fnmatch.translate(p)) for p in ignore_patterns]
            re_allow_pattern = [re.compile(fnmatch.translate(p)) for p in allow_patterns]

            expected_files = [f for f in filenames if not any(p.match(f) for p in re_ignore_pattern)]
            expected_files = [f for f in expected_files if any(p.match(f) for p in re_allow_pattern)]

            snapshot_folder = Path(pipeline_config_file).parent
            pipeline_is_cached = all((snapshot_folder / f).is_file() for f in expected_files)

            if pipeline_is_cached and not force_download:
                # if the pipeline is cached, we can directly return it
                # else call snapshot_download
                return snapshot_folder

        user_agent = {"pipeline_class": cls.__name__}
        if custom_pipeline is not None and not custom_pipeline.endswith(".py"):
            user_agent["custom_pipeline"] = custom_pipeline

        # download all allow_patterns - ignore_patterns
        try:
            cached_folder = snapshot_download(
                motion_module,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )

            return cached_folder

        except FileNotFoundError:
            # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
            # This can happen in two cases:
            # 1. If the user passed `local_files_only=True`                    => we raise the error directly
            # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
            if model_info_call_error is None:
                # 1. user passed `local_files_only=True`
                raise
            else:
                # 2. we forced `local_files_only=True` when `model_info` failed
                raise EnvironmentError(
                    f"Cannot load model {motion_module}: model is not cached locally and an error occured"
                    " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                    " above."
                ) from model_info_call_error

    @classmethod
    def get_pipeline_or_model_cls(
        cls, pipeline_cls_name: str, library_name: Optional[str] = None
    ) -> "AnimateDiffBasePipeline":
        if library_name is None:
            library_name = cls.__module__.split(".")[0]

        omni_vg_module = importlib.import_module(library_name)
        pipeline_cls = getattr(omni_vg_module, pipeline_cls_name)
        return pipeline_cls
