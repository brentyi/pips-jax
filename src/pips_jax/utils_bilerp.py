"""Utilities for bilinear interpolation."""


from typing import Tuple, Union

import jax
import jax.image
import jax.scipy.ndimage
from jax import numpy as jnp
from jaxtyping import Array, Float


def bilerp_coords_batched_hw(
    features: Float[Array, "*feat h w"],
    i: Float[Array, "*feat_then_coord"],
    j: Float[Array, "*feat_then_coord"],
) -> Float[Array, "*feat_then_coord"]:
    """Bilinear interpolation with leading batch axes."""
    h, w = features.shape[-2:]
    feat_batch = features.shape[:-2]
    assert i.shape[: len(feat_batch)] == j.shape[: len(feat_batch)] == feat_batch
    coord_batch = i.shape[len(feat_batch) :]

    def bilerp_single(
        features: Float[Array, "h w"],
        i: Float[Array, "*coord_batch"],
        j: Float[Array, "*coord_batch"],
    ) -> Array:
        assert features.shape == (h, w)
        assert i.shape == j.shape  # *coord_batch

        out = jax.scipy.ndimage.map_coordinates(
            features, (i, j), order=1, mode="constant"
        )
        return out

    # Vectorize over batch axes.
    bilerp_batched = bilerp_single
    for _ in range(len(feat_batch)):
        bilerp_batched = jax.vmap(bilerp_batched)

    out = bilerp_batched(features, i, j)
    assert out.shape == (*feat_batch, *coord_batch)
    return out


def bilerp_coords_batched_hwc(
    features: Float[Array, "*feat h w c"],
    i: Float[Array, "*feat_then_coord"],
    j: Float[Array, "*feat_then_coord"],
) -> Float[Array, "*feat_then_coord c"]:
    """Bilinear interpolation with leading batch axes and a trailing channel axis."""
    return jax.vmap(
        bilerp_coords_batched_hw,
        in_axes=(-1, None, None),
        out_axes=-1,
    )(features, i, j)


def resize_with_aligned_corners(
    image: jax.Array,
    shape: Tuple[int, ...],
    method: Union[str, jax.image.ResizeMethod],
    antialias: bool,
):
    """Alternative to jax.image.resize(), which emulates align_corners=True in PyTorch's
    interpolation functions."""
    spatial_dims = tuple(
        i
        for i in range(len(shape))
        if not jax.core.symbolic_equal_dim(image.shape[i], shape[i])
    )
    scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])
    translation = -(scale / 2.0 - 0.5)
    return jax.image.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )
