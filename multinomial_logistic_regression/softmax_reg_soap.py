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

import os
import sys
import fire
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('/home/ubuntu/polar/polargrad')

from polar_grad import PolarGrad
from muon import Muon_polar

sys.path.append('/home/ubuntu/soap/soap')

from soap import SOAP


def smooth(scalars: np.array, weight: float = 0.8) -> np.array:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

class MultinomialLogisticRegression(nn.Module):
    def __init__(self, d=100, K=10):
        super().__init__()
        self.W = nn.Parameter(torch.empty(d, K).uniform_(-1., 1.))

    def forward(self, x, y):
        # x \in (n, d)
        # W \in (d, K-1)
        scores = x @ self.W # (n, K)

        # Equiv to:
        # log_partition = torch.log(1 + torch.sum(torch.exp(scores), dim=1))
        zeros = torch.zeros(x.shape[0], 1, device=x.device)
        all_scores = torch.cat([zeros, scores], dim=1)  # (n, K)
        log_partition = torch.logsumexp(all_scores, dim=1)  # (n,)

        # w_{y_i}^T x_i
        label_scores = scores[torch.arange(x.shape[0]), y]  # (n,)

        return torch.sum(log_partition - label_scores)


def main(seed=42, steps=1500):
    # Check device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #device = torch.device("cpu")
    print(f"Using device: {device}")

    # Problem setup
    d, K = 100, 9
    N = 10000
    torch.manual_seed(seed)
    X = torch.randn(N, d, device=device)
    y = torch.randint(0, K, (N,), device=device)

    # Subsampling utility for mini-batch rows of X and y
    def sample_batch(batch_size=1000):
        idx = torch.randint(0, N, (batch_size,), device=device)
        return X[idx], y[idx]

    def run_stochastic_optimizer(optimizer_cls, method='qdwh', lr=5e-2, steps=steps, batch_size=1000, scheduler=False):
        torch.manual_seed(seed)
        model = MultinomialLogisticRegression(d, K).to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.)
        elif optimizer_cls == SOAP:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        else:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        losses = []
        condition_numbers_grad = []
        nuc_norms_grad = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
            X_batch, y_batch = sample_batch(batch_size)
            optimizer.zero_grad()
            loss = model(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
            condition_numbers_grad.append(torch.linalg.cond(model.W.grad).item())
            nuc_norms_grad.append(torch.linalg.matrix_norm(model.W.grad, ord='nuc').item())
        condition_numbers_grad = smooth(condition_numbers_grad, weight=0.8)
        nuc_norms_grad = smooth(nuc_norms_grad, weight=0.8)
        return losses, condition_numbers_grad, nuc_norms_grad


    # Compare optimizers
    loss_polar_grad, cond_grad_polar_grad, nuc_polar_grad = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=1e-4)
    loss_soap, cond_grad_soap, nuc_soap = run_stochastic_optimizer(SOAP, lr=3e-3)
    loss_muon, cond_grad_muon, nuc_muon = run_stochastic_optimizer(Muon_polar, method='ns', lr=7.5e-2)
    loss_muon_qdwh, cond_grad_muon_qdwh, nuc_muon_qdwh = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=7.5e-2)
    loss_muon_qdwh_decay, cond_grad_muon_qdwh_decay, nuc_muon_qdwh_decay = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=1.5e-1, scheduler=True)
    loss_adam, cond_grad_adam, nuc_adam = run_stochastic_optimizer(torch.optim.Adam, lr=5e-3)
    loss_adam_decay, cond_grad_adam_decay, nuc_adam_decay = run_stochastic_optimizer(torch.optim.Adam, lr=1e-2, scheduler=True)


    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad, label="PolarSGD (QDWH)", linestyle='-')
    axes[0].semilogy(loss_soap, label="SOAP", linestyle='--')
    axes[0].semilogy(loss_muon, label="Muon (NS)", linestyle='-.')
    axes[0].semilogy(loss_muon_qdwh, label="Muon (QDWH)", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_adam, label="Adam", linestyle='-')
    axes[0].semilogy(loss_adam_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k)$")

    # Plot condition numbers of gradients
    axes[1].plot(cond_grad_polar_grad, linestyle='-')
    axes[1].plot(cond_grad_soap, linestyle='--')
    axes[1].plot(cond_grad_muon, linestyle='-.')
    axes[1].plot(cond_grad_muon_qdwh, linestyle='-')
    axes[1].plot(cond_grad_muon_qdwh_decay, linestyle='--')
    axes[1].plot(cond_grad_adam, linestyle='-')
    axes[1].plot(cond_grad_adam_decay, linestyle='--')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla\mathsf{f}(X_k, \xi_k))$")
    
    # Plot nuclear norms of gradients
    axes[2].plot(nuc_polar_grad, linestyle='-')
    axes[2].plot(nuc_soap, linestyle='--')
    axes[2].plot(nuc_muon, linestyle='-.')
    axes[2].plot(nuc_muon_qdwh, linestyle='-')
    axes[2].plot(nuc_muon_qdwh_decay, linestyle='--')
    axes[2].plot(nuc_adam, linestyle='-')
    axes[2].plot(nuc_adam_decay, linestyle='--')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")

    fig.legend(loc='outside lower center', ncol=7, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f'fig/mat_log_reg_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)


    # Plot nuclear norms of gradients separately
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(nuc_polar_grad, linestyle='--')
    plt.plot(nuc_soap, linestyle='-.')
    plt.plot(nuc_muon, linestyle='-')
    plt.plot(nuc_muon_qdwh, linestyle='--')
    plt.plot(nuc_muon_qdwh_decay, linestyle='-.')
    plt.plot(nuc_adam, linestyle='-')
    plt.plot(nuc_adam_decay, linestyle='-.')
    plt.xlabel(r"iteration $k$")
    plt.ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")
    fig2.savefig(f'fig/mat_log_reg_nuc_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig2)


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
    
    fire.Fire(main)
