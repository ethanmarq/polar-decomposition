# PolarGrad Experiments

Empirical validation of the [PolarGrad](https://arxiv.org/abs/2505.21799) optimizer (Lau et al., 2025) across matrix optimization and language model pretraining benchmarks. PolarGrad applies polar decomposition (via QDWH or ZOLO-PD) as a gradient preconditioner, extending Muon with nuclear-norm scaling and without the need to tune Newton-Schulz polynomial coefficients.

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
  - [Qwen2.5 Pretraining](#qwen25-pretraining)
  - [Nonnegative Matrix Factorization](#nonnegative-matrix-factorization)
  - [Multi-Response Linear Regression](#multi-response-linear-regression)
  - [Multinomial Logistic Regression](#multinomial-logistic-regression)

---

## Hardware Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| GPU | A100 80GB (Ampere) | H100 SXM |
| GPU count | 1 (matrix experiments) | 1 (Qwen pretraining) |
| CUDA | 12.1+ | 12.6 |
| `bfloat16` | Required (Ampere+) | — |

> **Note:** V100 and earlier do not support native `bfloat16` hardware arithmetic. `torch.linalg.qr` (used in QDWH) does not support `bfloat16` on CUDA — PolarGrad upcasts gradients to `float32` internally at the top of each optimizer step and casts back before the weight update. The matrix optimization experiments run on CPU or single GPU.

---

## Installation

This project uses [pixi](https://prefix.dev/) for reproducible environment management.

**Install pixi:**

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**Clone the repository** (including the `polargrad` submodule):

```bash
git clone --recurse-submodules https://github.com/ethanmarq/polar-decomposition.git
```

If you already cloned without `--recurse-submodules`, initialize the submodule manually:

```bash
git submodule update --init --recursive
```

**Bootstrap the environment** (run once from the project root):

```bash
pixi install
```

All subsequent commands are run via `pixi run` — no manual `conda activate` or `pip install` needed.

---

## Repository Structure

```
polar-decomposition/
├── polargrad/                          # Core optimizer library
│   ├── polar_grad.py                   # PolarGrad optimizer (single-device)
│   ├── polar_grad_ddp.py               # PolarGrad with torch.distributed (DDP)
│   ├── polar.py                        # Polar decomposition dispatcher
│   ├── qdwh.py                         # QDWH algorithm
│   ├── zolopd.py                       # ZOLO-PD algorithm
│   ├── newton_schulz.py                # NS iteration (Muon-compatible)
│   ├── polar_express.py                # Polar Express (matrix-multiply only, bf16-safe)
│   └── muon.py                         # Muon baseline
├── qwen/                               # Qwen2.5 pretraining
├── nonnegative_matrix_factorization/   # NMF (two constraint implementations)
├── multi_response_linear_regression/   # Multi-response linear regression
├── multinomial_logistic_regression/    # Softmax / multinomial logistic regression
└── pixi.toml
```

---

## Experiments

All commands are run from the **project root** (`polar-decomposition/`).

---

### Qwen2.5 Pretraining

**Scripts:** `qwen/train_qwen.py`

**Objective:** Standard autoregressive cross-entropy over token sequences:

$$\mathcal{L}(\theta) = -\sum_{t} \log p_{\theta}(x_{t} \mid x_{\lt t})$$

PolarGrad or Muon is applied to 2D weight matrices; 1D parameters (biases, LayerNorm) are routed to AdamW.

```bash
# Full Qwen2.5 pretraining (A100/H100 required)
pixi run python -m qwen.train_qwen
```
---

### Nonnegative Matrix Factorization

**Scripts:** `nonnegative_matrix_factorization/nmf_s.py` · `nonnegative_matrix_factorization/nmf_np.py`

Given a nonnegative target matrix $M \in \mathbb{R}^{m \times n}$ (default: $m=500$, $n=250$, rank $r=5$) synthesized as $M = U_{\text{true}} V_{\text{true}}^\top$ with $U_{\text{true}}, V_{\text{true}} \geq 0$, find factors $X \in \mathbb{R}^{m \times r}$, $Y \in \mathbb{R}^{n \times r}$ with $X, Y \geq 0$ minimizing:

$$\mathcal{L}(X, Y) = \frac{1}{mn}\|XY^\top - M\|_F^2$$

The two scripts enforce the nonnegativity constraint differently:

#### Softplus Parameterization (`nmf_s.py`)

Nonnegativity is enforced via a smooth reparameterization. The raw parameters `self.X`, `self.Y` are unconstrained; the forward pass applies `softplus` before computing the loss:

```python
def forward(self, target):
    X = torch.nn.functional.softplus(self.X)
    Y = torch.nn.functional.softplus(self.Y)
    return torch.sum((X @ Y.T - target) ** 2) / target.numel()
```

The optimizer updates unconstrained variables, making the problem fully differentiable everywhere.

```bash
pixi run python -m nonnegative_matrix_factorization.nmf_s
```

#### Gradient Projection (`nmf_np.py`)

Nonnegativity is enforced by projecting parameters onto the nonnegative orthant after each optimizer step. The forward pass computes the loss directly on `self.X`, `self.Y`, and the constraint is applied post-update:

```python
def forward(self, target):
    return torch.sum((self.X @ self.Y.T - target) ** 2) / target.numel()

# After optimizer.step():
model.X.data.clamp_(min=0)
model.Y.data.clamp_(min=0)
```

```bash
pixi run python -m nonnegative_matrix_factorization.nmf_np
```

---

### Multi-Response Linear Regression

**Script:** `multi_response_linear_regression/multi_response_linear_reg.py`

**Objective:** Strongly convex matrix regression with deterministic gradients. Given fixed $A \in \mathbb{R}^{p \times m}$, $C \in \mathbb{R}^{p \times n}$ (default: $p=1000$, $m=500$, $n=100$), find $X \in \mathbb{R}^{m \times n}$ minimizing:

$$\mathcal{L}(X) = \frac{1}{2}\|AX - C\|_F^2$$

The closed-form gradient is $\nabla \mathcal{L}(X) = A^\top(AX - C)$ and the closed-form solution is $X^\star = (A^\top A)^{-1} A^\top C$. Plots track suboptimality $\mathcal{L}(X_k) - \mathcal{L}(X^\star)$, residual condition number $\kappa_2(AX_k - C)$, gradient condition number $\kappa_2(\nabla \mathcal{L}(X_k))$, and gradient nuclear norm.

```bash
pixi run python -m multi_response_linear_regression.multi_response_linear_reg
```

---

### Multinomial Logistic Regression

**Script:** `multinomial_logistic_regression/softmax_log_reg.py`

**Objective:** Strongly convex softmax regression with stochastic gradients. Given data $x \in \mathbb{R}^{N \times d}$, labels $y \in \{0,\ldots,K-1\}^N$, and weight matrix $W \in \mathbb{R}^{d \times K}$ (default: $d=100$, $K=9$, $N=10{,}000$), minimizes the cross-entropy loss:

$$\mathcal{L}(W) = \sum_{i=1}^{N} \left[\log (1 + \sum_{k=0}^{K} \exp \left(w_{k}^{T} x_{i})\right)) - (w_{y_i}^{T} x_i)\right]$$

A zero column is prepended to the scores before `logsumexp` for numerical stability (reference class convention). Mini-batches of size 1000 are drawn each step. Plots track training loss, gradient condition number $\kappa_2(\nabla_W \mathcal{L})$, and gradient nuclear norm.

```bash
pixi run python -m multinomial_logistic_regression.softmax_log_reg
```

---

## Citation

```bibtex
@article{lau2025polargrad,
  title={\textsc{PolarGrad}: A Class of Matrix-Gradient Optimizers from a Unifying Preconditioning Perspective},
  author={Lau, Tim Tsz-Kit and Qi Long and Weijie Su},
  year={2025},
  journal={arXiv preprint arXiv:2505.21799}
}
```
