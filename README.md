# DBVI-DGP

A clean, gpytorch-free implementation of **Diffusion Bridge Variational
Inference (DBVI)** for Deep Gaussian Processes (DGPs), as introduced in our paper:

> Jian Xu, Delu Zeng, Qibin Zhao, John Paisley.
> *Diffusion Bridge Variational Inference for Deep Gaussian Processes.*
> ICLR 2026. [OpenReview](https://openreview.net/forum?id=zyRmy0Ch9a)

---

## Method overview

For DGP inference with inducing variables $\mathbf U=\{U^{(\ell)}\}_{\ell=1}^L$,
classical mean-field VI (DSVI) approximates the posterior by a factorised Gaussian
$q(\mathbf U)=\prod_\ell \mathcal N(m_\ell, S_\ell)$. This assumption is restrictive — the
true posterior is non-Gaussian in deep models. Score-based DDVI relaxes the
Gaussian assumption but uses an **unconditional** noising process, leaving the
data dependence to enter only through the ELBO.

**DBVI** extends DDVI by introducing a **Doob h-transform** that constrains
the diffusion to interpolate between an amortized, data-anchored initial
distribution and a fixed terminal noise. The resulting **bridge SDE**
$$dU_t = \big[-\lambda(t)U_t + g(t)^2\,h(U_t,t,U_0)\big]\,dt + g(t)\,dW_t$$
keeps its marginal $p_t^{\text{Bri}}(U_t\mid x)=\mathcal N(m_t,\kappa_t I)$ Gaussian
(under affine forward drift; closed-form via Prop. 2 of the paper), but with
mean $m_t = \phi(t)\,\mu_\theta(x)$ now driven by an amortizer
$\mu_\theta(\cdot)$ that ingests the inducing-location context. A
**conditional score network** $s_\phi(U_t,t,\text{ctx})$ is trained against
this bridge marginal via denoising score matching:
$$\mathcal L_{\text{cond-DSM}}=\mathbb E_{t,U_t}\,\kappa_t\big\|s_\phi(U_t,t,\text{ctx}) + (U_t-m_t)/\kappa_t\big\|^2.$$
Posterior samples are drawn by integrating the **reverse-time bridge SDE**.

The Doob bridge therefore (i) injects data context directly into the
diffusion path, and (ii) bounds the path variance via the closed-form
$\kappa_t$ — two features unavailable to vanilla DDVI.

---

## Implementation notes

The original prototype distributed alongside the paper used `gpytorch` and
grafted the bridge SDE onto the standard `VariationalStrategy`. During
careful debugging we found that `gpytorch.variational._VariationalDistribution.initialize_variational_distribution`
overwrites the bridge-derived initial mean at first forward, which left the
score branch as an **isolated side-network** with no gradient path to the
ELBO. The DBVI claim is sound, but that early code did not faithfully
implement it.

**This repository contains a clean from-scratch implementation in which the
score network genuinely participates in the ELBO**, with the reverse-time
bridge SDE used as the sampling mechanism throughout training and
evaluation. The closed-form bridge marginals from Prop. 2 are pre-computed
on a 100-point time grid at model construction and used as targets for
conditional DSM.

The main script (`dbvi.py`) also includes several other variational families
(mean-field DSVI, plain DDVI, flow-based VI, IPVI, etc.) that share the
same DGP backbone for clean methodological comparison; the proper DBVI
variant is selected with `--variant dbvi-s`.

---

## Quick start

```bash
pip install torch pandas numpy tqdm
```

`dbvi.py` is a single-file script. The DBVI variant is `--variant dbvi-s`:

```bash
# DBVI (Doob-bridged score VI) on UCI energy
python dbvi.py --variant dbvi-s \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256 \
    --dsm_weight 1.0 \
    --doob_lambda 1.0 --doob_g 1.0 --doob_sigma0 1.0
```

Available datasets (in `data/`): `yacht`, `boston`, `energy`, `qsar`, `concrete`,
`power`, `protein` (standard UCI regression benchmarks).

A mean-field DSVI baseline for comparison:

```bash
python dbvi.py --variant dsvi \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --num_inducing 128 --batch_size 256
```

The unconditional DDVI variant (no bridge) is also available for ablation:

```bash
python dbvi.py --variant score \
    --dataset energy --data_path data/energy.csv \
    --epochs 100 --dsm_weight 1.0
```

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

Run `python dbvi.py --help` for the full list (also includes flags for the
ablation variants packaged in the same script).

---

## File map

```
dbvi.py               # main entry point — model + DBVI training + evaluation
aggregate_table.py    # builds RMSE / NLL summary tables across runs
data/                 # bundled UCI regression datasets
```

The DBVI-specific code lives in:

* `SparseGPLayer` — sparse GP layer with RBF–ARD kernel
* `Amortizer` — per-layer $\mu_\theta$ network mapping inducing context to vector
* `ConditionalScoreField` — conditional score network $s_\phi(U_t, t, \text{ctx})$
* `_precompute_doob` — numerically integrates Prop. 2 ODE for $\phi(t), \kappa(t)$
* `FlowDGP.sample_U` (`dbvi-s` branch) — reverse-bridge SDE sampler
* `FlowDGP.dsm_loss` (`dbvi-s` branch) — conditional DSM against bridge marginal
* `main()` — full DBVI training loop

---

## Citation

```bibtex
@inproceedings{
xu2026diffusion,
title={Diffusion Bridge Variational Inference for Deep Gaussian Processes},
author={JIAN XU and Delu Zeng and Qibin Zhao and John Paisley},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=zyRmy0Ch9a}
}
```

The DDVI predecessor:

```bibtex
@inproceedings{xu2024sparse,
  title={Sparse Inducing Points in Deep Gaussian Processes: Enhancing Modeling with Denoising Diffusion Variational Inference},
  author={Xu, Jian and Zeng, Delu and Paisley, John},
  booktitle={International Conference on Machine Learning},
  pages={55490--55500},
  year={2024},
  organization={PMLR}
}
```
