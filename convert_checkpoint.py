import re
from pathlib import Path
from typing import Mapping, Tuple, cast

import flax
import numpy as onp
import torch
import tyro
from einops import rearrange
from jax import numpy as jnp

from pips_jax.model import Pips


def main(
    torch_checkpoint_path: Path = Path("./torch_reference_model/model-000200000.pth"),
    jax_checkpoint_path: Path = Path("./checkpoints/reference_model/checkpoint_200000"),
    overwrite: bool = False,
) -> None:
    """Load a PyTorch PIPs checkpoint and convert it to a checkpoint for use in JAX."""

    # Load PyTorch parameters as numpy arrays.
    assert torch_checkpoint_path.exists()
    torch_state = torch.load(torch_checkpoint_path, map_location=torch.device("cpu"))
    torch_params = torch_state["model_state_dict"]
    torch_params = {k: v.numpy() for k, v in torch_params.items()}
    torch_step = int(torch_state["optimizer_state_dict"]["state"][0]["step"])
    print(f"Loaded PyTorch checkpoint at step {torch_step}!")
    print(
        f"{len(torch_params)} tensors, parameter count:",
        onp.sum([onp.prod(tensor.shape) for tensor in torch_params.values()]),
    )

    # Initialize JAX model.
    print("Initializing Flax model...")
    model = Pips(stride=4)
    params = model.init_params(seed=0)

    # Get flattened parameters.
    jax_params = cast(
        Mapping[Tuple[str, ...], jnp.ndarray],
        flax.traverse_util.flatten_dict(params),
    )
    print(
        f"{len(torch_params)} arrays, parameter count:",
        onp.sum([onp.prod(x.shape) for x in jax_params.values()]),
    )

    # Reconstruct the JAX parameter pytree using the PyTorch parameters.
    loaded_param_dict = {}
    for k, jax_param in jax_params.items():
        torch_name = ".".join(k[1:])
        torch_name = re.sub("kernel$", "weight", torch_name)
        torch_name = re.sub("scale$", "weight", torch_name)
        assert torch_name in torch_params

        # https://flax.readthedocs.io/en/latest/advanced_topics/convert_pytorch_to_flax.html
        loaded_param = torch_params[torch_name]
        if len(loaded_param.shape) == 2:
            loaded_param = rearrange(loaded_param, "o i -> i o")
        elif len(loaded_param.shape) == 3:
            loaded_param = rearrange(loaded_param, "o i 1 -> i o")
        elif len(loaded_param.shape) == 4:
            loaded_param = rearrange(loaded_param, "o i h w -> h w i o")

        assert loaded_param.shape == jax_param.shape
        loaded_param_dict[k] = loaded_param

    # Unflatten loaded parameters.
    loaded_param_dict = flax.traverse_util.unflatten_dict(loaded_param_dict)

    # Write checkpoint file.
    assert not jax_checkpoint_path.exists() or overwrite
    jax_checkpoint_path.write_bytes(flax.serialization.to_bytes(loaded_param_dict))
    print(f"Wrote checkpoint to {jax_checkpoint_path}!")


if __name__ == "__main__":
    tyro.cli(main)
