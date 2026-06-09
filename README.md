# ColorDecompose

Research codebase for **palette-based color decomposition** and **seamless merging of radiance fields**, built on top of [TensoRF: Tensorial Radiance Fields](https://apchenstu.github.io/TensoRF/) (ECCV 2022).

It extends TensoRF in two directions:

1. **Palette decomposition** — a compact color palette is extracted from the training images via RGB convex-hull simplification, and scene appearance is learned as barycentric mixtures over a *trainable* palette (`PLTRender`). Each rendered view can be split into per-palette color layers, and palette colors can be edited interactively at render time for scene recoloring. Optionally, a second palette over VGG semantic features is decomposed jointly (`MultiplePLTRender`).
2. **Scene merging / 3D Poisson editing** — two independently trained TensoRF scenes are composed (`ColorVMSplit`): a *target* object is placed into a *source* scene via a rigid transform, densities are max-blended per sample, and the seam is then blended by optimizing zero-initialized control copies of the target's appearance tensors and render MLP (ControlNet-style) with gradient-preservation losses, analogous to Poisson image editing in 3D.

## Installation

Tested with Python 3.10 + PyTorch + CUDA. A C compiler is required (Cython modules are compiled on first import via `pyximport`).

```bash
conda create -n ColorDecompose python=3.10
conda activate ColorDecompose
pip install torch torchvision
pip install tqdm scikit-image scikit-learn opencv-python configargparse einops \
    imageio imageio-ffmpeg tensorboard kornia lpips plyfile trimesh cvxopt \
    Cython matplotlib tkcolorpicker
# pytorch3d: follow https://github.com/facebookresearch/pytorch3d (must match your torch/CUDA build)
```

Note: `tkcolorpicker` (and a display) is needed for the interactive palette recoloring dialog that opens when rendering from a palette checkpoint.

## Datasets

* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
* [Synthetic-NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip)
* [Tanks&Temples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip)
* [Forward-facing (LLFF)](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
* Your own captures: calibrate with [instant-ngp's script](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) via `python dataLoader/colmap2nerf.py --colmap_matcher exhaustive --run_colmap`, then use `dataset_name = own_data` (see `configs/your_own_data.txt`).

Supported `dataset_name` values: `blender`, `llff`, `nsvf`, `tankstemple`, `own_data`, `blendermvs`. Dataset paths in the bundled configs point outside the repo — adjust `datadir` to your local layout.

## Quick start

All commands go through `main.py`, which dispatches to one of three subcommands: `train` (alias `test`), `merge`, and `buildcfg` (alias `cfg`).

### Train a scene

```bash
python main.py train --config configs/train/lego2.txt
```

Checkpoints and logs go to `log/<expname>/` (`<expname>.th`, TensorBoard events, periodic renders in `imgs_vis/`). Monitor with `tensorboard --logdir log`.

### Render / test

```bash
python main.py test --config configs/train/lego2.txt --render_only 1 --render_test 1
```

`--ckpt` defaults to `log/<expname>/<expname>.th`. Use `--render_train 1` / `--render_path 1` to render training views or a camera path. When the checkpoint contains a palette, a color-picker dialog opens per palette color — keep or change the colors to recolor the scene. Results are written to `log/<expname>/imgs_test_all/`, including per-palette layer images (`palette/`), depth maps (`rgbd/`), videos, and `mean.txt` (PSNR).

### Merge two trained scenes

The high-level driver trains both scenes if their checkpoints are missing, validates/expands the source AABB to cover the transformed target, generates the merge config plus a transforms JSON, and runs the merge:

```bash
python main.py buildcfg configs/gs/gxy3_source.txt configs/gs/gxy3_target.txt
```

Conventions: the **source** is the receiving scene (its dataset provides the cameras for evaluation); the **target** is the scene/object inserted into it, positioned by its rigid transform. The merged run lives in `log/<prefix>_merge/` (common name prefix of the two experiments) and writes a self-documenting config (`<prefix>_merge.txt` with the merge options plus commented `[Source]`/`[Target]` sections) along with `<prefix>_transforms.json`.

A merge can also be run directly from a config (see `configs/merge/*.txt`):

```bash
python main.py merge --config configs/merge/lego-over-ship.txt
```

`merge` requires `render_only = 1`. The composed model is initialized from `log/<expname>/<expname>.th` (a copy of the source scene's checkpoint — place it there when invoking `merge` manually), while `--ckpt` points to the target scene's checkpoint. Poisson blending iterations write intermediate renders to `imgs_test_iters/`; sampled-point caches are stored in `cache/` and reused across runs.

### Rigid transforms

Scene placement is specified by a JSON file passed as `--transform` (an explicit 4×4 `--matrix` is the mutually exclusive alternative). Keys are matched against experiment names after stripping their common prefix:

```json
{
    "source": {"rot": [1, 0, 0, 0], "trans": [0, 0, 0], "scale": [1, 1, 1]},
    "target": {"rot": [0.16, 0.88, 0.31, -0.31], "trans": [-0.51, -0.06, 0.0]}
}
```

`rot` is a quaternion in `(w, x, y, z)` order; scaling is applied before rotation/translation.

### Mesh / point-cloud export

```bash
python main.py train --config configs/train/lego2.txt --ckpt log/<expname>/<expname>.th --export_mesh 1
```

Exports a `.ply` mesh via marching cubes on the density field. Merge runs additionally export point clouds (`*_pc.ply`) and per-model meshes with `--export_mesh`.

## Configuration notes

Configs are plain-text `configargparse` files; any option can be overridden on the command line. Bundled sets:

* `configs/train/` — single-scene palette training
* `configs/merge/` — hand-written `X-over-Y` merge runs
* `configs/gs/` — source/target pairs used with `buildcfg`

Key options beyond standard TensoRF ones:

* `model_name` — `TensorVMSplit`, `TensorCP`, `TensorVM`, or `ColorVMSplit` (required for merging)
* `shadingMode` — `MLP_Fea`, `MLP_PE`, `SH`, … plus `PLT_Fea` (palette decomposition), `PLT_Fea_Multi` (RGB + semantic palettes), `PoissonMLPRender` (merge-capable MLP with control branch)
* `palette_type` — enable palette extraction/decomposition during training
* `semantic_type` — add VGG-feature semantics (PCA-projected) with a second palette
* `lossMode` — `PLTLoss`: reconstruction MSE + palette regularizers (`E_opaque`, convex-hull distance `PD`, `BLACK`)
* `transform` — rigid-transform JSON (see above); applied to camera poses at load time
* `at_least_aabb` — minimum bounding box the model must keep when shrinking its grid (set automatically by `buildcfg`)
* `n_lamb_sigma` / `n_lamb_sh`, `N_voxel_init` / `N_voxel_final`, `upsamp_list`, `update_AlphaMask_list` — tensor ranks and coarse-to-fine schedule, as in TensoRF

## Acknowledgements

* Built on the official [TensoRF](https://github.com/apchenstu/TensoRF) implementation by Anpei Chen et al.
* Palette extraction (`models/palette/`) adapts the RGB-space convex-hull palette decomposition code by Jianchao Tan et al. (*Decomposing Images into Layers via RGB-space Geometry*, TOG 2016, and follow-ups); point–triangle distance ported from [Geometric Tools](https://www.geometrictools.com/).

```
@INPROCEEDINGS{Chen2022ECCV,
  author = {Anpei Chen and Zexiang Xu and Andreas Geiger and Jingyi Yu and Hao Su},
  title = {TensoRF: Tensorial Radiance Fields},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2022}
}
```

## License

MIT — see [LICENSE](LICENSE).
