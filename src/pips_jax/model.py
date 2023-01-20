"""JAX port of the Persistent Independent Particles (PIPs) model.

https://github.com/aharley/pips
"""

from __future__ import annotations

import functools
from typing import Any, List, Literal, Tuple, cast

import flax
import flax.linen as nn
import jax
import jax_dataclasses as jdc
from einops import pack, rearrange, reduce, repeat
from jax import numpy as jnp
from jaxtyping import Array, Float, UInt8

from . import utils_bilerp, utils_pyramid

exact_gelu = functools.partial(nn.gelu, approximate=False)


@jdc.pytree_dataclass
class _TrackedParticles:
    """Internal state tracked by our PIPs model. Each particle is represented by a
    feature and a coordinate."""

    ffeats: Float[Array, "b s n c"]
    coords: Float[Array, "b s n 2"]


class Pips(nn.Module):
    S: int = 8
    stride: int = 8

    hidden_dim: int = 256
    latent_dim: int = 128

    corr_levels: int = 4
    corr_radius: int = 3

    @jdc.jit
    def init_params(
        self: jdc.Static[Pips], seed: int
    ) -> flax.core.FrozenDict[str, Any]:
        """Returns network parameters."""
        # Arbitary dimensions for dummy input.
        b = 1
        n = 3
        s = self.S
        h = 360
        w = 640

        xys = jnp.zeros((b, n, 2))
        rgbs = jnp.zeros((b, s, h, w, 3), dtype=jnp.uint8)
        return self.init(jax.random.PRNGKey(seed), xys, rgbs, iters=1, train=False)

    def setup(self):
        self.fnet = BasicEncoder(
            self.latent_dim,
            norm_fn="instance",
            dropout=0.0,
            stride=self.stride,
            name="fnet",  # type: ignore
        )
        self.delta_block = DeltaBlock()
        self.ffeat_updater = nn.Sequential(
            [
                nn.Dense(
                    self.latent_dim,
                    name="ffeat_updater.0",  # type: ignore
                ),
                exact_gelu,
            ]
        )
        torch_eps = 1e-5
        self.norm = nn.LayerNorm(  # This is a GroupNorm with 1 group in the original code.
            epsilon=torch_eps,
            reduction_axes=(-1,),
            name="norm",  # type: ignore
        )
        self.vis_predictor = nn.Sequential(
            [
                nn.Dense(
                    features=1,
                    name="vis_predictor.0",  # type: ignore
                )
            ]
        )

    def __call__(
        self,
        xys: Float[Array, "b n 2"],
        rgbs: UInt8[Array, "b s h w 3"],
        iters: int,
        train: bool,
    ) -> Tuple[Float[Array, "iters b s n 2"], Float[Array, "b s n"]]:
        """Run full PIPs model. Returns a tuple of arrays: coords at each iter,
        visibility logits."""

        b, n, _ = xys.shape
        assert rgbs.shape[:2] == (b, self.S)

        # Compute feature maps + initial features.
        fmap_pyramid, particles = self._initialize_tracking(xys, rgbs)

        # Iteratively refine particle coordinates & features.
        def update_step_wrapper(
            scope: Pips, carry_particles: _TrackedParticles
        ) -> Tuple[_TrackedParticles, Float[Array, "b s n 2"]]:
            """Wrapper for iterative refinement step, with a signature for use with
            `jax.lax.scan`."""
            updated = scope._iterative_update_step(fmap_pyramid, carry_particles, train)
            return updated, updated.coords * self.stride

        final_particles, coord_predictions = nn.scan(
            update_step_wrapper,
            variable_broadcast="params",
            split_rngs={"params": False, "dropout": True},
            length=iters,
        )(self, particles)

        # Extract visibility logits.
        vis_logits = self.vis_predictor(final_particles.ffeats)
        vis_logits = rearrange(vis_logits, "b s n 1 -> b s n", b=b, s=self.S, n=n)

        # Done!
        assert coord_predictions.shape == (iters, b, self.S, n, 2)
        assert vis_logits.shape == (b, self.S, n)
        return coord_predictions, vis_logits

    def _initialize_tracking(
        self,
        xys: Float[Array, "b n 2"],
        rgbs: UInt8[Array, "b s h w 3"],
    ) -> Tuple[utils_pyramid.FeaturePyramid, _TrackedParticles]:
        """Initialize feature pyramids and particles for tracking."""
        b, n, d = xys.shape
        assert d == 2

        b, s, h, w, c = rgbs.shape
        assert c == 3
        assert s == self.S

        hdown = h // self.stride
        wdown = w // self.stride

        # Initialize coordinates with zero velocity.
        coords_0 = xys / self.stride
        coords = repeat(coords_0, "b n d -> b s n d", s=s, d=2)
        assert coords.shape == (b, s, n, 2)

        # Create feature pyramid.
        rgbs = 2.0 * (rgbs / 255.0) - 1.0
        fmaps = self.fnet(rgbs)
        assert fmaps.shape == (b, s, hdown, wdown, self.latent_dim)
        fmap_pyramid = utils_pyramid.make_feature_pyramids(
            fmaps, num_levels=self.corr_levels
        )

        # Initialize features for the whole trajectory.
        ffeats = utils_bilerp.bilerp_coords_batched_hwc(
            fmaps[:, 0], i=coords[:, 0, :, 1], j=coords[:, 0, :, 0]
        )
        ffeats = repeat(ffeats, "b n c -> b s n c", s=self.S)
        assert ffeats.shape == (b, s, n, self.latent_dim)

        return fmap_pyramid, _TrackedParticles(ffeats, coords)

    def _iterative_update_step(
        self,
        fmap_pyramid: utils_pyramid.FeaturePyramid,
        state: _TrackedParticles,
        train: bool,
    ) -> _TrackedParticles:
        """Run a single iterative update step for our PIPs model."""

        b, s, n, c = state.ffeats.shape

        state = jdc.replace(state, coords=jax.lax.stop_gradient(state.coords))

        corr_pyramid = utils_pyramid.make_correlation_pyramids(
            fmap_pyramid, state.ffeats
        )
        assert len(corr_pyramid) == self.corr_levels

        # Only needed for training and visualization.
        #
        # fcp = jnp.zeros((b, s, n, hdown, wdown), dtype=jnp.float32)
        # for cr in range(self.corr_levels):
        #     fcp_ = corr_pyramid[cr]
        #     assert fcp_.shape[:3] == (b, s, n)
        #     assert fcp_.shape[3] <= hdown and fcp_.shape[4] <= wdown
        #     fcp_ = util_bilerp.resize_with_aligned_corners(
        #         fcp_,
        #         shape=(b, s, n, hdown, wdown),
        #         method=jax.image.ResizeMethod.LINEAR,
        #         antialias=False,
        #     )
        #     fcp = fcp + fcp_

        fcorrs = utils_pyramid.sample_correlation_features(
            corr_pyramid, state.coords, corr_radius=self.corr_radius
        )
        lww = fcorrs.shape[-1]  # lww = levels * window_dim * window_dim.
        assert fcorrs.shape == (b, s, n, lww)

        flows = state.coords - state.coords[:, 0:1, :, :]
        assert flows.shape == (b, s, n, 2)

        times = repeat(cast(Array, jnp.linspace(0, s, s)), "s -> b s n ()", b=b, n=n)

        # Reshape to b*n, s, 2 for MLP mixer.
        def rearrange_for_mixer(x: Array) -> Array:
            return rearrange(x, "b s n c -> (b n) s c", b=b, s=s, n=n)

        delta_all = self.delta_block(
            rearrange_for_mixer(state.ffeats),
            rearrange_for_mixer(fcorrs),
            rearrange_for_mixer(jnp.concatenate([flows, times], axis=-1)),
            train=train,
        )
        delta_all = rearrange(
            delta_all, "(b n) s c -> b s n c", b=b, s=s, n=n, c=2 + self.latent_dim
        )

        delta_coords = delta_all[:, :, :, :2]
        delta_feats = delta_all[:, :, :, 2:]
        delta_feats = self.ffeat_updater(self.norm(delta_feats))

        if not train:
            delta_coords = delta_coords.at[:, 0, :, :].set(0.0)

        return _TrackedParticles(
            state.ffeats + delta_feats, state.coords + delta_coords
        )


class BasicEncoder(nn.Module):
    """Basic encoder from RAFT."""

    output_dim: int
    norm_fn: Literal["instance"]
    dropout: float
    stride: int
    in_planes: int = 64

    @nn.compact
    def __call__(
        self, rgbs: Float[Array, "b s h w 3"]
    ) -> Float[Array, "... h w output_dim"]:
        assert self.norm_fn == "instance"  # Implemented with a LayerNorm.
        torch_eps = 1e-5

        b, s, h, w, d = rgbs.shape
        assert d == 3

        x = rgbs
        x = nn.Conv(
            self.in_planes,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=3,  # Note: an integer is needed for zero padding; not equivalent to "SAME".
            name="conv1",  # type: ignore
        )(x)
        x = nn.LayerNorm(
            epsilon=torch_eps,
            use_scale=False,
            use_bias=False,
            reduction_axes=(-3, -2),
        )(x)
        x = nn.relu(x)

        fmaps: List[Array] = []
        for i, (planes, stride) in enumerate(
            (
                (64, 1),
                (96, 2),
                (128, 2),
                (128, 2),
            )
        ):
            x = self._make_layer(planes, stride=stride, name=f"layer{i+1}")(x)
            assert x.shape[:2] == (b, s) and x.shape[-1] == planes
            fmaps.append(
                utils_bilerp.resize_with_aligned_corners(
                    x,
                    shape=(b, s, h // self.stride, w // self.stride, planes),
                    method=jax.image.ResizeMethod.LINEAR,
                    antialias=False,
                )
            )

        x, _ = pack(fmaps, "b s hdown wdown *")
        assert x.shape == (
            b,
            s,
            h // self.stride,
            w // self.stride,
            64 + 96 + 128 + 128,
        )

        x = nn.Conv(
            self.output_dim * 2,
            kernel_size=(3, 3),
            padding=1,
            name="conv2",  # type: ignore
        )(x)
        x = nn.LayerNorm(
            epsilon=torch_eps,
            use_scale=False,
            use_bias=False,
            reduction_axes=(-3, -2),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.output_dim,
            kernel_size=(1, 1),
            padding=0,
            name="conv3",  # type: ignore
        )(x)
        assert x.shape == (b, s, h // self.stride, w // self.stride, self.output_dim)

        # 1/11/2022: verified that everything up until here matches torch
        # implementation!
        return x

    def _make_layer(self, planes: int, stride: int, name: str) -> nn.Sequential:
        return nn.Sequential(
            [
                ResidualBlock(
                    planes=planes,
                    stride=stride,
                    name=f"{name}.0",  # type: ignore
                ),
                ResidualBlock(
                    planes=planes,
                    stride=1,
                    name=f"{name}.1",  # type: ignore
                ),
            ]
        )


class ResidualBlock(nn.Module):
    planes: int
    stride: int

    @nn.compact
    def __call__(
        self, x: Float[Array, "... h w c"]
    ) -> Float[Array, "... h_out w_out c"]:
        torch_eps = 1e-5

        skip = x
        x = nn.Conv(
            self.planes,
            kernel_size=(3, 3),
            strides=(self.stride, self.stride),
            padding=1,
            name="conv1",  # type: ignore
        )(x)
        x = nn.LayerNorm(
            epsilon=torch_eps,
            use_scale=False,
            use_bias=False,
            reduction_axes=(-3, -2),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            self.planes,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=1,
            name="conv2",  # type: ignore
        )(x)
        x = nn.LayerNorm(
            epsilon=torch_eps,
            use_scale=False,
            use_bias=False,
            reduction_axes=(-3, -2),
        )(x)
        x = nn.relu(x)

        if self.stride != 1:
            # Downsample for skip connection..
            skip = nn.Conv(
                self.planes,
                kernel_size=(1, 1),
                strides=(self.stride, self.stride),
                padding=0,
                name="downsample.0",  # type: ignore
            )(skip)
            skip = nn.LayerNorm(
                epsilon=torch_eps,
                use_scale=False,
                use_bias=False,
                reduction_axes=(-3, -2),
            )(skip)

        return nn.relu(x + skip)


class DeltaBlock(nn.Module):
    @nn.compact
    def __call__(
        self,
        ffeats: Float[Array, "bn s latent_dim"],
        fcorrs: Float[Array, "bn s lww"],
        flow_t: Float[Array, "bn s 3"],
        train: bool,
    ) -> Float[Array, "bn s latent_dim+2"]:
        bn, s, latent_dim = ffeats.shape
        assert fcorrs.shape[:2] == (bn, s)
        assert flow_t.shape[:2] == (bn, s)

        flow_t_sincos = get_3d_embedding(flow_t, c=64, cat_coords=True)
        x, _ = pack([ffeats, fcorrs, flow_t_sincos], "bn s *")
        x = MLPMixer(
            hidden_dim=512,
            output_dim=s * (latent_dim + 2),
            depth=12,
            expansion_factor=4,
            dropout=0.0,
            name="to_delta",  # type: ignore
        )(x, train=train)
        x = rearrange(x, "bn (s d) -> bn s d", bn=bn, s=s, d=latent_dim + 2)

        return x


class MLPMixer(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int
    expansion_factor: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self, x: Float[Array, "bn s input_dim"], train: bool
    ) -> Float[Array, "bn output_dim"]:
        x = nn.Dense(
            self.hidden_dim,
            name="0",  # type: ignore
        )(x)
        bn, s, hidden_dim = x.shape
        torch_eps = 1e-5

        for i in range(self.depth):

            # Token mixing with prenorm residual.
            skip = x
            x = nn.LayerNorm(
                epsilon=torch_eps,
                name=f"{i + 1}.0.norm",  # type: ignore
            )(x)
            x = rearrange(x, "bn s c -> bn c s", bn=bn, s=s, c=hidden_dim)
            x = self._feedforward(x, train, name=f"{i + 1}.0.fn")
            x = rearrange(x, "bn c s -> bn s c", bn=bn, s=s, c=hidden_dim)
            x = x + skip

            # Channel mixing with prenorm residual.
            #
            # Architectural nit: the layernorm scale/bias seems redundant given that
            # next layer is a linear one.
            skip = x
            x = nn.LayerNorm(
                epsilon=torch_eps,
                name=f"{i + 1}.1.norm",  # type: ignore
            )(x)
            x = self._feedforward(x, train, name=f"{i + 1}.1.fn")
            x = x + skip

        x = nn.LayerNorm(
            epsilon=torch_eps,
            name=str(1 + self.depth),  # type: ignore
        )(x)
        x = reduce(x, "bn s c -> bn c", reduction="mean", bn=bn, s=s, c=hidden_dim)
        x = nn.Dense(
            self.output_dim,
            name=str(3 + self.depth),  # type: ignore
        )(x)
        return x

    def _feedforward(self, x: Array, train: bool, name: str) -> Array:
        dim = x.shape[-1]
        x = nn.Sequential(
            [
                nn.Dense(dim * self.expansion_factor, name=f"{name}.0"),  # type: ignore
                exact_gelu,
                nn.Dropout(rate=self.dropout, deterministic=not train),
                nn.Dense(dim, name=f"{name}.3"),  # type: ignore
                nn.Dropout(self.dropout, deterministic=not train),
            ],
        )(x)
        return x


def get_3d_embedding(
    xyz: Float[Array, "b n 3"], c: int, cat_coords: bool = True
) -> Float[Array, "b n out_dim"]:
    """Transformer-style positional encoding, in 3D."""
    b, n, d = xyz.shape
    assert d == 3

    div_term = jnp.arange(0, c, 2) * (1000.0 / c)
    assert div_term.shape == (c // 2,)

    pe = repeat(xyz, "b n d -> b n d c", b=b, n=n, d=3, c=c)
    pe = pe * jnp.repeat(div_term, 2)
    pe = rearrange(pe, "b n d c -> b n (d c)", b=b, n=n, d=3, c=c)
    pe = pe.at[:, :, 1::2].add(jnp.pi / 2.0)
    pe = jnp.sin(pe)
    assert pe.shape == (b, n, c * 3)

    if cat_coords:
        pe = jnp.concatenate([pe, xyz], axis=2)
        assert pe.shape == (b, n, c * 3 + 3)

    return pe
