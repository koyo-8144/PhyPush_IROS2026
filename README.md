# PhyPush_IROS2026

![](imgs/phypush_pipeline_readme.png)


A Physics-guided Transformer that estimates an object's **mass** and **friction coefficient (μ)**
from a sequence of robotic end-effector velocity.

<!-- ## Repository Contents

| File | Purpose |
|------|---------|
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation / metrics + plots |
| `inspect_dataset.py` | Sanity-check the dataset: distributions + per-sample physics plots |
| `models.py` | `PhysicsTransformerEstimator` definition |
| `losses.py` | PINN physics losses + `log_mse_loss` |
| `configs.py` | All hyperparameters and dataset ranges |
| `utils.py` | Seeding, data reconstruction, helpers |
| `dataset.py` | **Required** — must provide `create_dataloaders` (see Note below) |

> **Note:** `train.py` and `evaluate.py` both `import dataset`. Make sure a `dataset.py`
> exposing `create_dataloaders(df, batch_size, m_seen_min, m_seen_max, mu_seen_min, mu_seen_max)`
> is present in the repo root before running. -->

## 1. Clone

```bash
git clone https://github.com/koyo-8144/PhyPush_IROS2026.git
cd PhyPush_IROS2026
```

## 2. Install

```bash
conda create -n phypush python=3.10 -y
conda activate phypush

pip install torch pandas numpy matplotlib scikit-learn ipython seaborn
```

For GPU support, install the CUDA build of PyTorch that matches your system from
https://pytorch.org/get-started/locally/.

## 3. Configure

All settings live in `configs.py` — there are **no command-line arguments**.

1. **Set the data path.** Edit `CSV_PATH` (selected via `FRAME_MODE`, `"local"` or `"world"`):

   ```python
   FRAME_MODE = "local"
   CSV_PATH = "/path/to/your/data.csv"
   ```

2. **Pick / tune a config.** Several presets are defined (`config_data`, `config_force`,
   `config_force_v2/v3/v4`). Choose which one is active at the bottom of the file:

   ```python
   used_config = config_force_v4
   ```

   <!-- Key fields you may want to change: `batch_size`, `num_epochs`, `init_lr`,
   `d_model`, `num_enc`, `transformer_ver`, the `pinn_coeffs` dict, and the
   seen/unseen mass & friction ranges (`M_SEEN_MAX`, `MU_SEEN_MAX`, etc.). -->

## 4. (Optional) Inspect the Dataset

Before training, you can sanity-check dataset:

```bash
python inspect_dataset.py
```

This uses the same `CSV_PATH` and ranges from `configs.py` and will:
- Print a macro summary and ground-truth statistics for `gt_mass` and `gt_mu`.
- Plot the global mass and friction distributions.
- For a few sample sequences, plot end-effector velocity, acceleration, force decomposition,
  a Newton's Second law check, and a Coulomb friction check.

Plots are shown interactively, so run it in an environment with a display
or an interactive backend.

## 5. Train

```bash
python train.py
```

This will:
- Load `CSV_PATH`, build train/val loaders, and initialize the model on GPU if available.
- Save outputs to `./results/checkpoints/from_20260618/<TIMESTAMP>/<MODEL_STRING>/`, including:
  - `config.json` — the exact config used
  - `best_model.pth` — best validation checkpoint
  - `transformer_epoch<N>.pth` — periodic checkpoints (every 200 epochs)
  - `training_progress.png` / `training_dynamics_full_report.png` — loss & estimation plots

Take note of the `<TIMESTAMP>` and `<MODEL_STRING>` printed during the run — you'll need them for evaluation.

## 6. Evaluate

Open `evaluate.py` and point it at the run you want to score:

```python
time  = "20260625_165857"          # the <TIMESTAMP> folder from training
model = "pinn_pcri-L1_p5c10.0"     # the <MODEL_STRING> folder from training
```

By default it loads `transformer_epoch1000.pth` from that folder
(change `WEIGHTS_PATH` for a different checkpoint, e.g. `best_model.pth`).
Then run:

```bash
python evaluate.py
```

This computes metrics and writes evaluation plots (e.g. `exp1_real_world_mass_accuracy.png`) into the same checkpoint directory.