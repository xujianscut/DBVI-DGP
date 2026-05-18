# Diffusion Bridge Variational Inference for Deep Gaussian Processes (DBVI-DGP)

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue)](https://openreview.net/forum?id=zyRmy0Ch9a)
[![OpenReview](https://img.shields.io/badge/OpenReview-zyRmy0Ch9a-orange)](https://openreview.net/forum?id=zyRmy0Ch9a)

Official implementation of the **ICLR 2026** paper
*"Diffusion Bridge Variational Inference for Deep Gaussian Processes"*.

DBVI extends DDVI by inserting a **Doob h-transform** into the forward
noising process so the diffusion **interpolates** between an amortized,
data-anchored initial distribution and a fixed terminal noise. The
**conditional score network** is then trained against the closed-form
bridge marginal, and posterior samples are drawn by integrating the
reverse-time bridge SDE.

---

## Method overview

For DGP inference with inducing variables $\mathbf U=\{U^{(\ell)}\}_{\ell=1}^L$,
mean-field VI (DSVI) assumes $q(\mathbf U)=\prod_\ell \mathcal N(m_\ell, S_\ell)$ —
too restrictive for deep models. Score-based DDVI relaxes the Gaussian assumption
but uses an **unconditional** noising process, leaving data dependence to enter
only through the ELBO.

DBVI introduces a **Doob bridge**:

$$dU_t = \big[-\lambda(t)U_t + g(t)^2\,h(U_t,t,U_0)\big]\,dt + g(t)\,dW_t$$

Under affine forward drift, the bridge marginal stays Gaussian (Prop. 2):
$p_t^{\text{Bri}}(U_t\mid x)=\mathcal N(m_t,\kappa_t I)$, with
$m_t = \phi(t)\,\mu_\theta(x)$ driven by an amortizer $\mu_\theta(\cdot)$ that
ingests the inducing-location context. A **conditional score network**
$s_\phi(U_t,t,\text{ctx})$ is trained against this marginal via conditional DSM:

$$\mathcal L_{\text{cond-DSM}}=\mathbb E_{t,U_t}\,\kappa_t\big\|s_\phi(U_t,t,\text{ctx}) + (U_t-m_t)/\kappa_t\big\|^2.$$

Posterior samples are drawn by integrating the **reverse-time bridge SDE**.
The Doob bridge therefore (i) injects data context directly into the
diffusion path, and (ii) bounds the path variance via the closed-form
$\kappa_t$ — both unavailable to vanilla DDVI.

---

## Repository layout

```
.
├── dbvi.py              Main entry — model + DBVI training + evaluation
├── aggregate_table.py   RMSE / NLL summary tables across runs
└── data/                Bundled UCI regression datasets
```

DBVI-specific components inside `dbvi.py`:

- `SparseGPLayer` — sparse GP layer with RBF–ARD kernel
- `Amortizer` — per-layer $\mu_\theta$ mapping inducing context to vector
- `ConditionalScoreField` — conditional score network $s_\phi(U_t, t, \text{ctx})$
- `_precompute_doob` — numerically integrates Prop. 2 ODE for $\phi(t), \kappa(t)$
- `FlowDGP.sample_U` (`dbvi-s` branch) — reverse-bridge SDE sampler
- `FlowDGP.dsm_loss` (`dbvi-s` branch) — conditional DSM against bridge marginal
- `main()` — full training loop

---

## Requirements

```bash
pip install torch pandas numpy tqdm
```

---

## Quick start

`dbvi.py` is a single-file script. Select the DBVI variant with `--variant dbvi-s`:

```bash
# DBVI (Doob-bridged score VI) on UCI energy
python dbvi.py --variant dbvi-s \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 \
    --dsm_weight 1.0 \
    --doob_lambda 1.0 --doob_g 1.0 --doob_sigma0 1.0
```

Available datasets (in `data/`): `yacht`, `boston`, `energy`, `qsar`,
`concrete`, `power`, `protein` (standard UCI regression benchmarks).

Baselines packaged in the same script:

```bash
# Mean-field DSVI
python dbvi.py --variant dsvi --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256

# Unconditional DDVI (no bridge) — for ablation
python dbvi.py --variant score --dataset energy --data_path data/energy.csv \
    --epochs 100 --dsm_weight 1.0
```

---

## Implementation notes

**This repository contains a clean from-scratch implementation in which the
score network genuinely participates in the ELBO**, with the reverse-time
bridge SDE used as the sampling mechanism throughout training and evaluation.
The closed-form bridge marginals from Prop. 2 are pre-computed on a 100-point
time grid at model construction and used as targets for conditional DSM.

`dbvi.py` also includes several other variational families (mean-field DSVI,
plain DDVI, flow-based VI, IPVI, etc.) sharing the same DGP backbone for clean
methodological comparison; the proper DBVI variant is `--variant dbvi-s`.

---

## Key DBVI-specific arguments

| Flag | Default | Purpose |
|---|---|---|
| `--variant dbvi-s` | — | Select Doob-bridged DBVI |
| `--dsm_weight` | 0.0 | Coefficient on conditional DSM; **set ≥1.0 for proper DBVI training** |
| `--dsm_samples` | 1 | $(t,\varepsilon)$ MC samples per layer per minibatch |
| `--doob_lambda` | 1.0 | $\lambda$ of forward bridge drift $f=-\lambda U$ |
| `--doob_g` | 1.0 | $g$ of forward bridge diffusion |
| `--doob_sigma0` | 1.0 | Initial std $\sigma_0$ of $p_0^\theta(U_0\mid x)$ |
| `--flow_steps` | 10 | Reverse-bridge-SDE integration steps |
| `--flow_hidden` | 128 | Width of amortizer / score network |
| `--num_inducing` | 128 | $M$ per layer |
| `--layers` | 2 | DGP depth |
| `--mc_samples` | 2 | MC samples for ELBO data term |
| `--eval_samples` | 32 | MC samples at evaluation |

Run `python dbvi.py --help` for the full list.

---

## Citation

```bibtex
@inproceedings{xu2026diffusion,
  title     = {Diffusion Bridge Variational Inference for Deep Gaussian Processes},
  author    = {Xu, Jian and Zeng, Delu and Zhao, Qibin and Paisley, John},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=zyRmy0Ch9a}
}
```

The DDVI predecessor:

```bibtex
@inproceedings{xu2024sparse,
  title     = {Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author    = {Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  pages     = {55490--55500},
  year      = {2024},
  publisher = {PMLR}
}
```

---

## Contact

Questions / issues: open a GitHub issue or contact the corresponding author
(see paper).
