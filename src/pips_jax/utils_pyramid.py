"""Utilities for working with feature pyramids."""

from typing import List, NewType, cast

from einops import rearrange
from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float

from . import utils_bilerp

FeaturePyramid = NewType(
    "FeaturePyramid", List[Float[Array, "b s h_level w_level dim"]]
)
CorrelationPyramid = NewType("CorrelationPyramid", List[Float[Array, "b s n h w"]])


def make_feature_pyramids(
    fmaps: Float[Array, "b s h w dim"], num_levels: int
) -> FeaturePyramid:
    """Make a multi-resolution feature pyramid."""
    b, s, h, w, latent_dim = fmaps.shape

    fmap_pyramid = [fmaps]
    for i in range(num_levels - 1):
        fmap_pyramid.append(
            nn.avg_pool(fmap_pyramid[-1], window_shape=(2, 2), strides=(2, 2))
        )
    return FeaturePyramid(fmap_pyramid)


def make_correlation_pyramids(
    fmap_pyramid: FeaturePyramid, targets: Float[Array, "b s n dim"]
) -> CorrelationPyramid:
    """Compute a correlation pyramid from a feature pyramid and target features."""
    b, s, n, dim = targets.shape

    corr_pyramid = []
    for fmaps in fmap_pyramid:
        _, fmaps_s, h, w, fmaps_dim = fmaps.shape
        assert s == fmaps_s
        assert dim == fmaps_dim
        corr = jnp.einsum("bshwd,bsnd->bsnhw", fmaps, targets) / jnp.sqrt(dim)
        assert corr.shape == (b, s, n, h, w)
        corr_pyramid.append(corr)
    return CorrelationPyramid(corr_pyramid)


def sample_correlation_features(
    corr_pyramid: CorrelationPyramid,
    coords: Float[Array, "b s n 2"],
    corr_radius: int,
) -> Float[Array, "b s n lww"]:
    """Sample a multi-resolution correlation feature from a correlation pyramid."""
    b, s, n, d = coords.shape
    assert d == 2

    window_dim = 2 * corr_radius + 1
    delta = cast(Array, jnp.mgrid[:window_dim, :window_dim]) - corr_radius

    out_pyramid = []
    for i in range(len(corr_pyramid)):
        corrs = corr_pyramid[i]

        coords_lvl = (coords / (2**i))[..., None, None] + delta
        assert coords_lvl.shape == (b, s, n, 2, window_dim, window_dim)

        corrs = utils_bilerp.bilerp_coords_batched_hw(
            corrs, i=coords_lvl[:, :, :, 1], j=coords_lvl[:, :, :, 0]
        )
        assert corrs.shape == (b, s, n, window_dim, window_dim)

        corrs = rearrange(corrs, "b s n w1 w2 -> b s n (w1 w2)")
        out_pyramid.append(corrs)

    return rearrange(out_pyramid, "l b s n ww -> b s n (l ww)")
