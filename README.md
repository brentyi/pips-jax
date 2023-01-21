# pips-jax

JAX port of the [PIPs model](https://github.com/aharley/pips) for tracking point
trajectories.

```
@inproceedings{harley2022particle,
  title={Particle Video Revisited: Tracking Through Occlusions Using Point Trajectories},
  author={Adam W Harley and Zhaoyuan Fang and Katerina Fragkiadaki},
  booktitle={ECCV},
  year={2022}
}
```

We currently include:

- Model implementation using [flax](https://github.com/google/flax).
- JAX version of PIPs's 12/15/22 reference checkpoint.
- PyTorch -> JAX checkpoint translation script.

### Setup

Clone and install (you may want to
[install JAX with GPU support](https://github.com/google/jax#pip-installation-gpu-cuda)
first):

```
git clone https://github.com/brentyi/pips-jax.git
cd pips-jax
pip install -e .
```

Un-split reference checkpoint:

```
# Full checkpoints surpass GitHub's maximum file size, so we split the reference
# checkpoint into several parts.
cat checkpoints/reference_model/checkpoint_200000.* > checkpoints/reference_model/checkpoint_200000
```

Runnable scripts:

- `python convert_checkpoint.py --help`: Conversion script for converting the
  PIPs reference PyTorch checkpoint for use in Flax.
- `python demo.py --help`: Loose reproduction of the original PIPs model's demo
  script. Loads images and writes GIFs:

  ![demo_image_000](./demo_out/demo_000.gif)

- `python benchmark.py --help`: Benchmarking script for the JAX model's forward
  pass. Results on a single forward pass[^1]:

  ```
  JAX (jitted):
    0.10610 ± 0.00165 seconds

  PyTorch (with torch.inference_mode):
    0.17770 ± 0.01157 seconds
  ```

[^1]:
    Run on an RTX 2080 TI. 8 image subsequence, 640x360, 256 points, stride=4,
    iters=6.

### Progress

- [x] Check model outputs against torch
- [x] Checkpoint translation
- [x] Reproduce demo script
- [x] Benchmark
- [ ] Double-check install from scratch
