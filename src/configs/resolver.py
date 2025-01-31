"""Custom OmegaConf resolvers used by Hydra configs in this repository."""

import os

import torch
from omegaconf import OmegaConf


# undefined paths default to . (current directory)
def register_resolvers():
    """
    Registers custom resolvers for OmegaConf, enabling dynamic configuration values.
    The following resolvers are registered:
    - `cwd`: Resolves to the current working directory using `os.getcwd`.
    - `home`, `work`, `scratch`: Resolves to the corresponding environment variables (`HOME`, `WORK`, `SCRATCH`),
      defaulting to "." if the variable is not set.
    - `gpu_count`: Resolves to the number of available GPUs using `torch.cuda.device_count`.
    - `node_count`: Resolves to the number of nodes in a SLURM job using the `SLURM_JOB_NUM_NODES` environment variable,
      defaulting to 1 if the variable is not set.
    - `cpu_count`: Resolves to the number of CPUs available using `os.cpu_count`.
    - `env`: Resolves to the value of a specified environment variable, defaulting to "." if the variable is not set.

    Example usage:
    ```yaml
    path: ${scratch:}/data
    gpu_count: ${gpu_count:}
    ```
    """

    if not OmegaConf.has_resolver("cwd"):
        OmegaConf.register_new_resolver("cwd", os.getcwd)
        for env_var in ["HOME", "WORK", "SCRATCH"]:
            OmegaConf.register_new_resolver(env_var.lower(), lambda: os.environ.get(env_var, "."))

        # Others
        OmegaConf.register_new_resolver("gpu_count", torch.cuda.device_count)
        OmegaConf.register_new_resolver("cpu_count", os.cpu_count)
        OmegaConf.register_new_resolver("min", lambda x, y: min(x, y))  # e.g. ${min:2, ${cpu_count:}}
        OmegaConf.register_new_resolver("add", lambda x, y: x + y)  # e.g. ${add:2, ${cpu_count:}}
        OmegaConf.register_new_resolver("env", lambda key: os.environ.get(key, "."))
