# Copyright 2025 Tim Tsz-Kit Lau.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import sys
import os
import fire
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

import torch
import torch.nn as nn

sys.path.append('/home/ethanm/repos/polar-decomposition/polargrad')

from polar_grad import PolarGrad
from muon import Muon_polar


def smooth(scalars: np.array, weight: float = 0.9) -> np.array:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

# Factorization model: X = A @ B^T
class NMFModel(nn.Module):
    def __init__(self, m=500, n=250, r=5):
        super().__init__()
        # Optimizer lives in unconstrained space
        self.X = nn.Parameter(torch.empty(m, r).uniform_(0., 1.))
        self.Y = nn.Parameter(torch.empty(n, r).uniform_(0., 1.))

    def forward(self, target):
        return torch.sum((self.X @ self.Y.T - target) ** 2) / target.numel()

class NMFModelAltGD(nn.Module):
    def __init__(self, m=500, n=250, r=5, lr=1e-2):
        super().__init__()
        self.m = m
        self.n = n
        self.r = r
        self.lr = lr

        # Initialize X and Y
        self.X = nn.Parameter(torch.empty(m, r).uniform_(-1., 1.))
        self.Y = nn.Parameter(torch.empty(n, r).uniform_(-1., 1.))

    def loss(self, target):
        mse_loss = torch.sum((self.X @ self.Y.T - target) ** 2) / target.numel()
        return mse_loss

    def alternating_gradient_step(self, target, num_inner_steps=1):
        N = target.numel()

        L_X = (2.0 / N) * torch.linalg.matrix_norm(self.Y, ord=2)**2
        lr_X = 1.0 / L_X
        for _ in range(num_inner_steps):
            loss_X = self.loss(target)
            grad_X = torch.autograd.grad(loss_X, self.X, retain_graph=True)[0]
            self.X.data = self.X.data - lr_X * grad_X
            self.X.data.clamp_(min=0)

        # Y-update with X fixed: L_Y = (2/N) * ||X||_2^2
        L_Y = (2.0 / N) * torch.linalg.matrix_norm(self.X, ord=2)**2
        lr_Y = 1.0 / L_Y
        for _ in range(num_inner_steps):
            loss_Y = self.loss(target)
            grad_Y = torch.autograd.grad(loss_Y, self.Y, retain_graph=False)[0]
            self.Y.data = self.Y.data - lr_Y * grad_Y
            self.Y.data.clamp_(min=0)

        # In alternating_gradient_step, print every 50 iters:
        if not hasattr(self, '_step'): self._step = 0
        self._step += 1
        if self._step % 50 == 0:
            sigma_Y = torch.linalg.matrix_norm(self.Y, ord=2).item()
            sigma_X = torch.linalg.matrix_norm(self.X, ord=2).item()
            print(f"step {self._step}: σ₁(Y)={sigma_Y:.3f}, σ₁(X)={sigma_X:.3f}, "
                f"lr_X={lr_X.item():.2f}, lr_Y={lr_Y.item():.2f}, "
                f"loss={self.loss(target).item():.4e}")

        return (torch.linalg.cond(grad_X), torch.linalg.cond(grad_Y),
                torch.linalg.matrix_norm(grad_X, ord='nuc'),
                torch.linalg.matrix_norm(grad_Y, ord='nuc'))

    # def alternating_gradient_step(self, target, num_inner_steps=1):
    #     # Update X while fixing Y
    #     for _ in range(num_inner_steps):
    #         loss_X = self.loss(target)
    #         grad_X = torch.autograd.grad(loss_X, self.X, retain_graph=True)[0]
    #         self.lr = 1 / torch.linalg.norm(grad_Y, ord='fro')**2
    #         self.X.data = self.X.data - self.lr * grad_X
    #         self.X.data.clamp_(min=0)

    #     # Update Y while fixing X
    #     for _ in range(num_inner_steps):
    #         loss_Y = self.loss(target)
    #         grad_Y = torch.autograd.grad(loss_Y, self.Y, retain_graph=False)[0]
    #         self.lr = 1 / torch.linalg.norm(grad_X, ord='fro')**2
    #         self.Y.data = self.Y.data - self.lr * grad_Y
    #         self.Y.data.clamp_(min=0)

    #     return torch.linalg.cond(grad_X), torch.linalg.cond(grad_Y), torch.linalg.matrix_norm(grad_X, ord='nuc'), torch.linalg.matrix_norm(grad_Y, ord='nuc')

    def fit(self, target, steps=1000, num_inner_steps=1):
        losses = []
        condition_numbers_grad_X = []
        condition_numbers_grad_Y = []
        nuc_norms_grad_X = []
        nuc_norms_grad_Y = []
        for i in tqdm(range(steps), desc=f"optimizer = AltGD"):
            cond_grad_X, cond_grad_Y, nuc_grad_X, nuc_grad_Y = self.alternating_gradient_step(target, num_inner_steps=num_inner_steps)
            current_loss = self.loss(target)
            losses.append(current_loss.item())
            condition_numbers_grad_X.append(cond_grad_X.item())
            condition_numbers_grad_Y.append(cond_grad_Y.item())
            nuc_norms_grad_X.append(nuc_grad_X.item())
            nuc_norms_grad_Y.append(nuc_grad_Y.item())
        condition_numbers_grad_X = smooth(condition_numbers_grad_X)
        condition_numbers_grad_Y = smooth(condition_numbers_grad_Y)
        return losses, condition_numbers_grad_X, condition_numbers_grad_Y, nuc_norms_grad_X, nuc_norms_grad_Y

# # class LowRankModel(nn.Module):
# #     def __init__(self, m=500, n=250, r=5):
#         super().__init__()
#         self.X = nn.Parameter(torch.empty(m, r).uniform_(-1., 1.))
#         self.Y = nn.Parameter(torch.empty(n, r).uniform_(-1., 1.))
#
#     def forward(self, target, mask):
#         mse_loss = torch.sum((self.X @ self.Y.T - target) ** 2) / mask.sum()
#         return mse_loss

# class LowRankModelAltGD(nn.Module):
#     def __init__(self, m=500, n=250, r=5, lr=1e-2):
#         """
#         Low-Rank Matrix Factorization Model: M ≈ U V^T
#
#         Args:
#             m: number of rows of M
#             n: number of columns of M
#             r: target rank
#             weight_decay: regularization strength (default 0)
#         """
#         super().__init__()
#         self.m = m
#         self.n = n
#         self.r = r
#         self.lr = lr
#
#         # Initialize X and Y
#         self.X = nn.Parameter(torch.empty(m, r).uniform_(-1., 1.))
#         self.Y = nn.Parameter(torch.empty(n, r).uniform_(-1., 1.))
#
#     def loss(self, target, mask):
#         """
#         Computes the masked loss: squared error only over observed entries.
#
#         Args:
#             target: observed matrix (m, n)
#             mask: binary mask (m, n), 1 if observed, 0 if missing
#         Returns:
#             scalar loss        """
#         mse_loss = torch.sum((self.X @ self.Y.T - target) ** 2) / mask.sum()
#         return mse_loss
#
#     def alternating_gradient_step(self, target, mask, num_inner_steps=1):
#         """Performs one step of masked alternating gradient descent."""
#         # Update X while fixing Y
#         for _ in range(num_inner_steps):
#             loss_X = self.loss(target, mask)
#             grad_X = torch.autograd.grad(loss_X, self.X, retain_graph=True)[0]
#             self.X.data = self.X.data - self.lr * grad_X
#
#         # Update Y while fixing X
#         for _ in range(num_inner_steps):
#             loss_Y = self.loss(target, mask)
#             grad_Y = torch.autograd.grad(loss_Y, self.Y, retain_graph=False)[0]
#             self.Y.data = self.Y.data - self.lr * grad_Y
#
#         return torch.linalg.cond(grad_X), torch.linalg.cond(grad_Y), torch.linalg.matrix_norm(grad_X, ord='nuc'), torch.linalg.matrix_norm(grad_Y, ord='nuc')
    #
    # def fit(self, target, mask, steps=1000, num_inner_steps=1):
    #     """
    #     Fit the model to observed entries using masked alternating minimization.
    #
    #     Args:
    #         target: observed matrix (m, n)
    #         mask: binary mask (m, n)
    #         steps: number of alternating minimization steps
    #         num_inner_steps: number of least-squares solves per U/V update
    #     """
    #     losses = []
    #     condition_numbers_grad_X = []
    #     condition_numbers_grad_Y = []
    #     nuc_norms_grad_X = []
    #     nuc_norms_grad_Y = []
    #     for i in tqdm(range(steps), desc=f"optimizer = AltGD"):
    #         cond_grad_X, cond_grad_Y, nuc_grad_X, nuc_grad_Y = self.alternating_gradient_step(target, mask, num_inner_steps=num_inner_steps)
    #         current_loss = self.loss(target, mask)
    #         losses.append(current_loss.item())
    #         condition_numbers_grad_X.append(cond_grad_X.item())
    #         condition_numbers_grad_Y.append(cond_grad_Y.item())
    #         nuc_norms_grad_X.append(nuc_grad_X.item())
    #         nuc_norms_grad_Y.append(nuc_grad_Y.item())
    #     condition_numbers_grad_X = smooth(condition_numbers_grad_X)
    #     condition_numbers_grad_Y = smooth(condition_numbers_grad_Y)
    #     return losses, condition_numbers_grad_X, condition_numbers_grad_Y, nuc_norms_grad_X, nuc_norms_grad_Y


def main(seed=42, steps=1000):
    # Check device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    m, n, r = 500, 250, 5
    U_true = torch.abs(torch.randn(m, r, device=device))
    V_true = torch.abs(torch.randn(n, r, device=device))
    M = U_true @ V_true.T  # Ground truth low-rank matrix

    # Observed entries mask (simulate missing data)
    # mask = (torch.rand(m, n, device=device) < 0.3).float()
    #
    # Training loop
    def train_lowrank(optimizer_cls, method='qdwh', lr=0.1, steps=steps, scheduler=False):
        torch.manual_seed(seed)
        model = NMFModel()
        model = model.to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.)
        else:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        losses = []
        condition_numbers_grad_X = []
        condition_numbers_grad_Y = []
        nuc_norms_grad_X = []
        nuc_norms_grad_Y = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
            loss = model(M)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.X.data.clamp_(min=0)
            model.Y.data.clamp_(min=0)
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
            condition_numbers_grad_X.append(torch.linalg.cond(model.X.grad).item())
            condition_numbers_grad_Y.append(torch.linalg.cond(model.Y.grad).item())
            nuc_norms_grad_X.append(torch.linalg.matrix_norm(model.X.grad, ord='nuc').item())
            nuc_norms_grad_Y.append(torch.linalg.matrix_norm(model.Y.grad, ord='nuc').item())
        condition_numbers_grad_X = smooth(condition_numbers_grad_X)
        condition_numbers_grad_Y = smooth(condition_numbers_grad_Y)
        return losses, condition_numbers_grad_X, condition_numbers_grad_Y, nuc_norms_grad_X, nuc_norms_grad_Y

    # Compare optimizers
    torch.manual_seed(seed)
    loss_altgd, cond_X_altgd, cond_Y_altgd, nuc_X_altgd, nuc_Y_altgd = NMFModelAltGD(lr=5e1).to(device).fit(M, steps=steps)

    loss_polar_grad_lr, cond_X_polar_grad_lr, cond_Y_polar_grad_lr, nuc_X_polar_grad_lr, nuc_Y_polar_grad_lr = train_lowrank(PolarGrad, method='qdwh', lr=3.162e+1)
    loss_polar_grad_lr_decay, cond_X_polar_grad_lr_decay, cond_Y_polar_grad_lr_decay, nuc_X_polar_grad_lr_decay, nuc_Y_polar_grad_lr_decay = train_lowrank(PolarGrad, method='qdwh', lr=1.778e+01, scheduler=True)
    loss_muon_qdwh_lr, cond_X_muon_qdwh_lr, cond_Y_muon_qdwh_lr, nuc_X_muon_qdwh_lr, nuc_Y_muon_qdwh_lr  = train_lowrank(Muon_polar, method='qdwh', lr=3.162e-1) # 5e-1
    loss_muon_qdwh_lr_decay, cond_X_muon_qdwh_lr_decay, cond_Y_muon_qdwh_lr_decay, nuc_X_muon_qdwh_lr_decay, nuc_Y_muon_qdwh_lr_decay = train_lowrank(Muon_polar, method='qdwh', lr=5.623e-1, scheduler=True)
    loss_muon_ns_lr, cond_X_muon_ns_lr, cond_Y_muon_ns_lr, nuc_X_muon_ns_lr, nuc_Y_muon_ns_lr  = train_lowrank(Muon_polar, method='ns', lr=3.162e-1)
    loss_adam_lr, cond_X_adam_lr, cond_Y_adam_lr, nuc_X_adam_lr, nuc_Y_adam_lr = train_lowrank(torch.optim.Adam, lr=1.000e-01) # 5e-3
    loss_adam_lr_decay, cond_X_adam_lr_decay, cond_Y_adam_lr_decay, nuc_X_adam_lr_decay, nuc_Y_adam_lr_decay = train_lowrank(torch.optim.Adam, lr=5.623e-1, scheduler=True)


    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad_lr, label="PolarGrad (QDWH)", linestyle='-')
    axes[0].semilogy(loss_polar_grad_lr_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_muon_ns_lr, label="Muon (NS)", linestyle='-.')
    axes[0].semilogy(loss_muon_qdwh_lr, label="Muon (QDWH)", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_lr_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_adam_lr, label="Adam", linestyle='-')
    axes[0].semilogy(loss_adam_lr_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_altgd, label="AltGD", linestyle=':', color='red')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k,Y_k)$")

    # Plot condition numbers of gradients of X
    axes[1].plot(cond_X_polar_grad_lr, linestyle='-')
    axes[1].plot(cond_X_polar_grad_lr_decay, linestyle='--')
    axes[1].plot(cond_X_muon_ns_lr, linestyle='-.')
    axes[1].plot(cond_X_muon_qdwh_lr, linestyle='-')
    axes[1].plot(cond_X_muon_qdwh_lr_decay, linestyle='--')
    axes[1].plot(cond_X_adam_lr, linestyle='-')
    axes[1].plot(cond_X_adam_lr_decay, linestyle='--')
    axes[1].plot(cond_X_altgd, linestyle=':')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla_X \mathsf{f}(X_k, Y_k))$")

    # Plot condition numbers of gradients of Y
    axes[2].plot(cond_Y_polar_grad_lr, linestyle='-')
    axes[2].plot(cond_Y_polar_grad_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_muon_ns_lr, linestyle='-.')
    axes[2].plot(cond_Y_muon_qdwh_lr, linestyle='-')
    axes[2].plot(cond_Y_muon_qdwh_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_adam_lr, linestyle='-')
    axes[2].plot(cond_Y_adam_lr_decay, linestyle='--')
    # axes[2].plot(cond_Y_altgd, linestyle=':')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\kappa_2(\nabla_Y \mathsf{f}(X_k, Y_k))$")
    
    fig.legend(loc='outside lower center', ncol=8, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f'fig/nonneg_mat_factorization_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)

    # # Plots without Adam
    # fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    # axes[0].semilogy(loss_polar_grad_lr, label="PolarGrad (QDWH)", linestyle='-')
    # axes[0].semilogy(loss_polar_grad_lr_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    # axes[0].semilogy(loss_muon_ns_lr, label="Muon (NS)", linestyle='-.')
    # axes[0].semilogy(loss_muon_qdwh_lr, label="Muon (QDWH)", linestyle='-')
    # axes[0].semilogy(loss_muon_qdwh_lr_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    # axes[0].semilogy(loss_adam_lr_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    # # axes[0].semilogy(loss_altgd, label="AltGD", linestyle=':')
    # axes[0].set_xlabel(r"iteration $k$")
    # axes[0].set_ylabel(r"$\mathsf{f}(X_k,Y_k)$")
    #
    # # Plot condition numbers of gradients of X
    # axes[1].plot(cond_X_polar_grad_lr, linestyle='-')
    # axes[1].plot(cond_X_polar_grad_lr_decay, linestyle='--')
    # axes[1].plot(cond_X_muon_ns_lr, linestyle='-.')
    # axes[1].plot(cond_X_muon_qdwh_lr, linestyle='-')
    # axes[1].plot(cond_X_muon_qdwh_lr_decay, linestyle='--')
    # axes[1].plot(cond_X_adam_lr_decay, linestyle='--')
    # # axes[1].plot(cond_X_altgd, linestyle=':')
    # axes[1].set_xlabel(r"iteration $k$")
    # axes[1].set_ylabel(r"$\kappa_2(\nabla_X \mathsf{f}(X_k, Y_k))$")
    #
    # # Plot condition numbers of gradients of Y
    # axes[2].plot(cond_Y_polar_grad_lr, linestyle='-')
    # axes[2].plot(cond_Y_polar_grad_lr_decay, linestyle='--')
    # axes[2].plot(cond_Y_muon_ns_lr, linestyle='-.')
    # axes[2].plot(cond_Y_muon_qdwh_lr, linestyle='-')
    # axes[2].plot(cond_Y_muon_qdwh_lr_decay, linestyle='--')
    # axes[2].plot(cond_Y_adam_lr_decay, linestyle='--')
    # # axes[2].plot(cond_Y_altgd, linestyle=':')
    # axes[2].set_xlabel(r"iteration $k$")
    # axes[2].set_ylabel(r"$\kappa_2(\nabla_Y \mathsf{f}(X_k, Y_k))$")
    #
    # fig.legend(loc='outside lower center', ncol=6, bbox_to_anchor=(0.51, -0.05), borderaxespad=0., fontsize=16)
    # fig.subplots_adjust(bottom=0.15)
    # fig.savefig(f'fig/low_rank_mat_comp_2_{seed}.pdf', dpi=500, bbox_inches='tight')
    # plt.close(fig)
    #
    #
    # fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    # # Plot nuclear norms of gradients of X
    # axes2[0].plot(nuc_X_polar_grad_lr[0:200], label="PolarGrad (QDWH)", linestyle='-')
    # axes2[0].plot(nuc_X_polar_grad_lr_decay[0:200], label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    # # axes2[0].plot(nuc_X_muon_ns_lr[0:200], label="Muon (NS)", linestyle='-.')
    # axes2[0].plot(nuc_X_muon_qdwh_lr[0:200], label="Muon (QDWH)", linestyle='-')
    # axes2[0].plot(nuc_X_muon_qdwh_lr_decay[0:200], label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    # axes2[0].plot(nuc_X_adam_lr[0:200], label="Adam", linestyle='-')
    # axes2[0].plot(nuc_X_adam_lr_decay[0:200], label=r"Adam (lr $\downarrow$)", linestyle='--')
    # #axes2[0].plot(nuc_X_altgd[0:200], label="AltGD", linestyle=':')
    # axes2[0].set_xlabel(r"iteration $k$")
    # axes2[0].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla_X \mathsf{f}(X_k, Y_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")
    #
    # # Plot nuclear norms of gradients of Y
    # axes2[1].plot(nuc_Y_polar_grad_lr[0:200], linestyle='-')
    # axes2[1].plot(nuc_Y_polar_grad_lr_decay[0:200], linestyle='--')
    # # axes2[1].plot(nuc_Y_muon_ns_lr[0:200], linestyle='-.')
    # axes2[1].plot(nuc_Y_muon_qdwh_lr[0:200], linestyle='-')
    # axes2[1].plot(nuc_Y_muon_qdwh_lr_decay[0:200], linestyle='--')
    # axes2[1].plot(nuc_Y_adam_lr[0:200], linestyle='-')
    # axes2[1].plot(nuc_Y_adam_lr_decay[0:200], linestyle='--')
    # #axes2[1].plot(nuc_Y_altgd[0:200], linestyle=':')
    # axes2[1].set_xlabel(r"iteration $k$")
    # axes2[1].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla_Y \mathsf{f}(X_k, Y_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")
    #
    # fig2.legend(loc='outside lower center', ncol=7, bbox_to_anchor=(0.51, -0.05), borderaxespad=0., fontsize=16)
    # fig2.subplots_adjust(bottom=0.15)
    # fig2.savefig(f'fig/low_rank_mat_comp_3_{seed}.pdf', dpi=500, bbox_inches='tight')
    # plt.close(fig2)
import json

def train_one_run(method_key, lr, M, seed, steps, scheduler=False, device='cpu'):
    """Run one training trajectory. Returns the loss list. Bails early on NaN/Inf."""
    torch.manual_seed(seed)

    if method_key == 'altgd':
        model = NMFModelAltGD(lr=lr).to(device)
        losses, *_ = model.fit(M, steps=steps)
        return losses

    model = NMFModel().to(device)
    if method_key == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    elif method_key == 'polar_grad':
        opt = PolarGrad(model.parameters(), method='qdwh', lr=lr, momentum=0.)
    elif method_key == 'muon_qdwh':
        opt = Muon_polar(model.parameters(), method='qdwh', lr=lr)
    elif method_key == 'muon_ns':
        opt = Muon_polar(model.parameters(), method='ns', lr=lr)
    else:
        raise ValueError(f"Unknown method_key: {method_key}")

    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.95) if scheduler else None

    losses = []
    for _ in range(steps):
        loss = model(M)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.X.data.clamp_(min=0)
        model.Y.data.clamp_(min=0)
        if sched: sched.step()
        losses.append(loss.item())
        if not np.isfinite(losses[-1]):
            break  # diverged
    return losses


def evaluate_lr(method_key, lr, M, seeds, steps, scheduler=False, device='cpu'):
    """Mean of last-10% loss across seeds. Returns inf if any seed diverged or loss didn't decrease."""
    tail_metrics = []
    for seed in seeds:
        losses = np.asarray(train_one_run(method_key, lr, M, seed, steps, scheduler, device))
        if len(losses) < steps or not np.all(np.isfinite(losses)) or losses[-1] >= losses[0]:
            return float('inf')
        tail_metrics.append(float(np.mean(losses[int(0.9 * len(losses)):])))
    return float(np.mean(tail_metrics))


def sweep_lr(method_key, M, seeds, steps, scheduler=False, coarse_lrs=None, refine=True, device='cpu'):
    """Coarse log sweep then a refined sweep half a decade around the best."""
    if coarse_lrs is None:
        coarse_lrs = np.logspace(-4, 2, 13)

    label = f"{method_key}{' (lr↓)' if scheduler else ''}"
    results = {}
    print(f"\n=== {label}: coarse sweep ===")
    for lr in coarse_lrs:
        metric = evaluate_lr(method_key, float(lr), M, seeds, steps, scheduler, device)
        results[float(lr)] = metric
        tag = "" if np.isfinite(metric) else "  [diverged]"
        print(f"  lr={float(lr):.3e}  -> {metric:.4e}{tag}")

    finite = {lr: v for lr, v in results.items() if np.isfinite(v)}
    if not finite:
        print(f"  WARNING: every lr failed for {label}; widen the range")
        return None, results

    best_coarse = min(finite, key=finite.get)
    if refine:
        print(f"  refining around lr={best_coarse:.3e}")
        for lr in np.logspace(np.log10(best_coarse) - 0.5, np.log10(best_coarse) + 0.5, 5):
            if float(lr) in results:
                continue
            metric = evaluate_lr(method_key, float(lr), M, seeds, steps, scheduler, device)
            results[float(lr)] = metric
            tag = "" if np.isfinite(metric) else "  [diverged]"
            print(f"  lr={float(lr):.3e}  -> {metric:.4e}{tag}")
        finite = {lr: v for lr, v in results.items() if np.isfinite(v)}

    best_lr = min(finite, key=finite.get)
    print(f"  --> best {label}: lr={best_lr:.3e}  (loss={finite[best_lr]:.4e})")
    return best_lr, results


def tune_lr(steps=1000, n_seeds=3, output='lr_tune.json'):
    """Sweep all optimizer configs across seeds and save best lrs to JSON."""
    device = torch.device("cpu")

    # Fixed data seed so the target M is the same across all runs
    torch.manual_seed(42)
    m, n, r = 500, 250, 5
    U_true = torch.abs(torch.randn(m, r, device=device))
    V_true = torch.abs(torch.randn(n, r, device=device))
    M = U_true @ V_true.T

    seeds = list(range(n_seeds))

    # Per-optimizer starting ranges, chosen by how each method scales the update.
    # Wide enough to bracket your current hand-tuned values; narrow if too slow.
    configs = [
        # (json_key,         method_key,    scheduler, coarse grid)
        # ('altgd',            'altgd',       False, np.logspace( 0, 3, 7)),
        ('polar_grad',       'polar_grad',  False, np.logspace(-1, 2, 7)),
        ('polar_grad_decay', 'polar_grad',  True,  np.logspace(-1, 2, 7)),
        ('muon_qdwh',        'muon_qdwh',   False, np.logspace(-3, 1, 9)),
        ('muon_qdwh_decay',  'muon_qdwh',   True,  np.logspace(-3, 1, 9)),
        ('muon_ns',          'muon_ns',     False, np.logspace(-3, 1, 9)),
        ('adam',             'adam',        False, np.logspace(-4, 0, 9)),
        ('adam_decay',       'adam',        True,  np.logspace(-4, 0, 9)),
    ]

    best, all_runs = {}, {}
    for key, method_key, sched, coarse in configs:
        best_lr, runs = sweep_lr(method_key, M, seeds, steps,
                                 scheduler=sched, coarse_lrs=coarse, device=device)
        best[key] = best_lr
        all_runs[key] = {f"{lr:.6e}": v for lr, v in runs.items()}

    with open(output, 'w') as f:
        json.dump({'best_lrs': best, 'all_results': all_runs,
                   'config': {'steps': steps, 'n_seeds': n_seeds, 'seeds': seeds}},
                  f, indent=2)

    print("\n=== Summary ===")
    for k, v in best.items():
        print(f"  {k}: {v:.3e}" if v is not None else f"  {k}: FAILED")
    print(f"\nWrote {output}")

if __name__ == "__main__":
    if not os.path.exists('fig'):
        os.makedirs('fig')
    # Default settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(['science', 'grid', 'notebook'])
    # These are the colors that will be used in the plot
    tab10_colors = list(plt.get_cmap('tab10').colors)     # 10 colors
    dark2_colors = plt.get_cmap('Dark2').colors           # 8 colors

    # Pick two distinct additions
    additional_colors = [dark2_colors[3], dark2_colors[5]]

    # Combine to make 12-color palette
    color_sequence = tab10_colors + additional_colors

    plt.rcParams.update({
        "text.usetex": True,
        "axes.prop_cycle": plt.cycler(color=color_sequence),
        } 
        )
    
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(precision=8)
    
    fire.Fire({'main': main, 'tune_lr': tune_lr})
