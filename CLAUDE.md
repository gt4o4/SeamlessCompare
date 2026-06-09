# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase (PyTorch, CUDA) built on top of TensoRF (tensorial radiance fields). It extends TensoRF in two directions:

1. **Palette-based color decomposition**: scene appearance is expressed as barycentric mixtures of a small, trainable color palette extracted from the training images via RGB convex-hull simplification.
2. **Scene merging / 3D Poisson editing**: two independently trained TensoRF checkpoints are merged (source placed into target via a rigid transform), then seams are blended by optimizing zero-initialized "control" copies of the appearance tensors (ControlNet-style) with gradient-preservation losses.

The top-level `README.md` is mostly the upstream TensoRF readme; its commands (`python train.py --config ...`) are stale — `train.py` has no `__main__`. The real entry point is `main.py`.

## Commands

```bash
# Train a single scene (configs in configs/train/, configs/gs/, etc.)
python main.py train --config configs/train/lego2.txt

# Render from an existing checkpoint ("test" is an alias for "train")
python main.py train --config configs/train/lego2.txt --render_only 1 --render_test 1 [--ckpt path/to.th]

# Merge a trained source into a trained target (requires render_only; configs in configs/merge/)
python main.py merge --config configs/merge/lego-over-ship.txt

# End-to-end orchestration: trains either scene if its ckpt is missing, computes/validates
# AABBs, generates the merge config + transforms JSON, then runs the merge ("cfg" alias)
python main.py buildcfg configs/gs/gxy3_source.txt configs/gs/gxy3_target.txt

# Monitor training
tensorboard --logdir log
```

- Outputs land in `./log/{expname}/`: checkpoint `{expname}.th`, TensorBoard events, `imgs_vis/`, `imgs_test_all/` (with `rgbd/`, `palette/` layer images, `*video.mp4`, `mean.txt` PSNR), merge caches in `cache/*.npy`.
- There is no test suite or linter. `models/palette/test.py` is a standalone palette-extraction demo, not a unit test. `extra/auto_run_paramsets.py` is a legacy param-sweep runner.
- Environment: Python 3.10, PyTorch + CUDA. Beyond the README's pip list, the code imports **pytorch3d** (knn/ball_query), **trimesh**, **einops**, **configargparse**, **cvxopt**, **scikit-learn**, **plyfile**, **tkcolorpicker**, and **Cython** — `.pyx` modules are compiled at import time via `pyximport` (needs a C compiler).
- Datasets are referenced by relative/absolute paths in the config files (e.g. `../nerf-pytorch/data/nerf_synthetic/lego`); they live outside the repo.

## Architecture

### Entry point and config system

`main.py` defines `ConfigParser` (subcommand dispatch) and `SetupEnvironment` (cudaMallocAsync, TF32, cudnn.benchmark, pyximport install). Each subcommand maps to a `ConfigCommand` class: `train.py`, `merge.py`, `buildcfg.py`. Each `ConfigCommand.__init__` declares configargparse arguments; `__call__(args)` runs the job and returns the `Trainer`/`Evaluator`/`Merger`.

Config values like `model_name`, `shadingMode`, `lossMode` are parsed **directly into class factories** through the `MODEL_ZOO` / `RENDER_ZOO` / `LOSS_ZOO` registries in `models/__init__.py` (`ClassCollection` resolves class names or `_aliases`, e.g. `PLT_Fea` → `PLTRender`, `MLP_Fea` → `MLPRender_Fea`). So `args.model_name(...)` instantiates the model class.

`buildcfg.py` is config metaprogramming: it re-uses the train/merge parsers to round-trip configs (`get_parser_cfg`/`build_args_command` in `main.py` serialize a parsed namespace back to config text), finds the newest `**/{expname}.th` checkpoint, transforms the target's AABB corners to derive the source's `at_least_aabb`, writes a per-scene transforms JSON, and emits a self-documenting merge config (Merge + commented Source/Target sections) into the merge log folder.

### Models (`models/`)

- `tensorBase.py` `TensorBase`: ray sampling/marching, alpha-grid mask, AABB shrink, `forward` composites `n_dim`-channel render output (3 for plain RGB; `3 + len(palette) - 1` when a palette render module is used).
- `tensoRF.py`: `TensorVM`, `TensorCP`, `TensorVMSplit` decompositions (density + appearance planes/lines, `basis_mat`).
- `renderBase.py` `PLTRender`: MLP outputs per-palette logits; `rgb_from_palette_rev` converts them to barycentric weights over the **trainable** `palette` parameter; returns `[rgb, opaque]` concatenated. `MultiplePLTRender` stacks one `PLTRender` per palette (`PLT_NAMES = ('RGB', 'SEM')`) for joint RGB + semantic decomposition.
- `colorRF.py` `ColorVMSplit` (the merge-capable model): keeps a list of `merge_target` models. Source-vs-target coordinate mapping is the inverse 4×4 `--matrix` applied in `MultipleGridMask.shift_and_scale`. `compute_validmask` returns a per-model **bitmask**; `feature2density` max-blends densities (scaled by `render_gap` / `density_gain`) and records which model won per sample (`NormalizeCoord.idx`); `compute_radiance` then routes each sample's RGB to the winning model. `enable_trainable_control` clones `app_plane/app_line/basis_mat` into `*_ctl` copies. `PoissonMLPRender.enable_trainable_control` appends a zero-init `nn.Linear(3,3)` to a copy of the MLP; the control output is added via a forward hook.
- `models/loss.py` `PLTLoss`: reconstruction MSE plus regularizers weighted by `RegWeights_t`: `E_opaque` (push opacities to 1), `PD` (keep the trainable palette inside the original RGB convex hull, via Delaunay + Cython point-triangle distance), `BLACK` (norm of the last palette color).
- `models/palette/`: convex-hull palette extraction (`Hull_Simplification_determined_version`), ported research code; `GteDistPointTriangle.pyx` is Cython compiled on first import.
- `TransformFile` (`models/__init__.py`): JSON of named rigid transforms (`rot` quaternion, `trans`, `scale`). Lookup strips the common prefix of the two experiment names, a `_VM` suffix, and maps `scene`→`scan`.

### Training (`train.py`)

`Trainer.reconstruction`: coarse-to-fine voxel upsampling (`upsamp_list`, log-spaced `N_voxel_init→N_voxel_final`), alpha-mask updates + AABB shrink + ray filtering (`update_AlphaMask_list`), TV/L1/Ortho regularizers, periodic eval + checkpoint save. `build_palette` runs hull simplification on training-image RGBs (foreground only when `white_bg`); `build_sem_palette` does the same on VGG16+PCA semantic features when `semantic_type` is set (`dataLoader/semantic_helper.py`). 4-channel GT images are composited against a *random* white/black background per pixel during training.

### Merging (`merge.py`)

`Merger` (subclass of `Evaluator`): target model is loaded from `{basedir}/{expname}.th`, source from `--ckpt`; calls `add_merge_target`, exports point cloud/mesh, samples and caches visible points + gradients (`cache/*.npy` memmaps, regenerated if stale), then `poisson_editing` optimizes only the control copies with `loss_diff` (preserve RGB gradients across the seam) and `loss_pin` (pin source-region colors), writing intermediate renders to `imgs_test_iters/`.

### Data (`dataLoader/`)

`dataset_dict` = blender / llff / tankstemple / nsvf / own_data / blendermvs. Loaders accept `transform_type` (rigid transform applied to camera poses at load time, from `TransformFile`) and `semantic_type`/`pca` for semantic features.

## Gotchas

- `Trainer.build_network` calls `pick_palette()` (a tkinter color-picker GUI) whenever a checkpoint with a palette is loaded — this blocks/fails on headless machines.
- `merge` asserts `render_only = 1` in its config.
- Checkpoints embed model-constructor `kwargs`; loading is `args.model_name(**ckpt['kwargs'])` then `tensorf.load(ckpt)` — model class changes must stay compatible with stored kwargs.
- `PoissonMLPRender.__init__` checks a hardcoded `aval_rep.npy` path and, if it exists, silently switches view directions to KNN lookups from that file.
- Much of the code uses dense, idiosyncratic Python (walrus operators, `zip_longest` with `super()` as fillvalue, `UserString`/`UserDict` subclasses, config text round-tripping). Match the existing style and be careful with seemingly-redundant constructs — they're usually load-bearing.
