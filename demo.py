from pathlib import Path

import cv2
import flax
import imageio.v3 as iio
import jax
import numpy as onp
import tyro
from einops import rearrange
from jaxtyping import Array, Float

from pips_jax.model import Pips


def main(
    checkpoint_path: Path = Path("./checkpoints/reference_model/checkpoint_200000"),
    demo_image_dir: Path = Path("./demo_images/"),
    demo_image_extension: str = "jpg",
    gif_out_dir: Path = Path("./demo_out"),
    sqrt_n: int = 16,
) -> None:
    """Load a set of images, track n points on subsequences of them, and save
    visualizations as gifs."""

    # Initialize model, load params from reference checkpoint.
    print("Initializing model...")
    model = Pips(stride=4)
    params = model.init_params(seed=0)
    params = flax.serialization.from_bytes(params, checkpoint_path.read_bytes())

    @jax.jit
    def run_model(
        xys: Float[Array, "b n 2"], rgbs: Float[Array, "b s h w 3"]
    ) -> Float[Array, "b s n 2"]:
        """Run the PIPs model, and return coordinates at the final iteration of refinement."""
        coord_preds, vis_logits = model.apply(
            params, xys=xys, rgbs=rgbs, iters=6, train=False
        )
        assert coord_preds.shape == (
            6,
            *rgbs.shape[:2],
            sqrt_n**2,
            2,
        )  # (iters, b, s, n, 2)
        return coord_preds[-1]

    S = model.S
    image_paths = sorted(demo_image_dir.glob(f"*.{demo_image_extension}"))
    max_iters = len(image_paths) // S
    print(f"Found {len(image_paths)} images, set {max_iters=}.")

    for global_step in range(max_iters):
        # Read images.
        images = []
        for s in range(S):
            fn = image_paths[global_step * S + s]
            print(fn)

            image = iio.imread(fn)
            image = image.astype(onp.uint8)
            assert image.shape == (360, 640, 3)
            assert image.dtype == onp.uint8
            images.append(image)

        rgbs = rearrange(images, "s h w c -> 1 s h w c")
        s, h, w, c = rgbs.shape[1:]
        assert c == 3

        # Initialize points to track, of shape (b, n, 2). Each point should be in the form (x, y).
        coords_0 = rearrange(
            onp.mgrid[:sqrt_n, :sqrt_n], "d i j -> j i d", i=sqrt_n, j=sqrt_n, d=2
        )
        coords_0 = coords_0 / (sqrt_n - 1.0)
        coords_0 = (
            # Note: original code doesn't subtract 1 here.
            coords_0 * (onp.array([w, h]) - 1.0 - 16.0)
            + 8.0
        )
        coords_0 = rearrange(coords_0, "w h d -> 1 (w h) d", h=sqrt_n, w=sqrt_n, d=2)

        # Track points. OpenCV visualization expects integers.
        tracked_points = onp.array(run_model(coords_0, rgbs)).astype(onp.int64)
        n = sqrt_n**2
        assert tracked_points.shape == (1, S, n, 2)

        # Visualize correspondences. There's probably fancier code we could copy from
        # elsewhere for this.
        vis_rgbs = []
        colors = onp.random.randint(low=128, high=256, size=(n, 3))
        for s in range(S):
            vis_rgb = images[s]

            # Draw tracks.
            for i in range(n):
                for ss in range(1, s + 1):
                    cv2.line(
                        vis_rgb,
                        *tracked_points[0, ss - 1 : ss + 1, i],
                        color=(colors[i] * ss // (s + 1)).tolist(),
                        thickness=2,
                    )

            # Draw circles.
            for i in range(n):
                cv2.circle(
                    vis_rgb,
                    tracked_points[0, s, i],
                    radius=3,
                    color=colors[i].tolist(),
                    thickness=-1,
                )

            vis_rgbs.append(vis_rgb)

        # Write GIF.
        target_path = gif_out_dir / f"demo_{global_step:03d}.gif"
        gif_out_dir.mkdir(parents=True, exist_ok=True)
        iio.imwrite(target_path, onp.array(vis_rgbs), loop=0)


if __name__ == "__main__":
    tyro.cli(main)
