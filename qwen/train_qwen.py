import os
import math
import torch
import json
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

from typing import Tuple, Optional

def polar(a: torch.Tensor,
        *,
        method: str = 'qdwh',
        compute_hermitian: bool = False,
        eps: Optional[float] = None,
        max_iterations: Optional[int] = None, 
        ns_coeffs: tuple = (3.4445, -4.7750, 2.0315)) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the polar decomposition.

    Given the m x n matrix `a`, returns the factors of the polar
    decomposition ``u`` (also m x n) and ``h`` such that
    ``a = u h`` (if m >= n; ``h`` is n x n) or
    ``a = h u`` (if m < n; ``h`` is m x m), where ``h`` is symmetric (Hermitian) positive semidefinite.
    If `a` is nonsingular, ``h`` is positive definite and the decomposition is unique.
    The unitary factor ``u`` has orthonormal columns unless n > m, in which case it has orthonormal rows.

    Three methods are supported:

      * ``method="ns"``:
        Applies the Newton-Schulz iteration to compute the polar decomposition.
        This method requires the choice of three coefficients ``a``, ``b``, and ``c``
        of the iterating matrix polynomial
      * ``method="qdwh"``:
        Applies the QDWH (QR-based Dynamically Weighted Halley) algorithm.
        This method is more efficient for large matrices and is numerically stable
      * ``method="zolo-pd"``:
        Applies the ZOLO-PD (Zolotarev-based Polar Decomposition) algorithm.
        This method is more efficient for large matrices and is numerically stable
    Args:
        a: A full-rank input matrix of shape (m, n). The matrix may be padded if it
        represents a smaller true shape.
        method: Either "qdwh" or "svd" (default "qdwh").
        compute_hermitian: If True, the Hermitian positive-semidefinite factor is computed.
        eps: The precision tolerance; if None, the machine epsilon for `a.dtype` is used.
        max_iterations: Maximum iterations for QDWH. Ignored if ``method != "qdwh"``.
                    If None, a default (e.g. 10) is used.

    Returns:
        A tuple ``(unitary, posdef)`` where:
        - ``unitary`` is the computed unitary factor (m x n),
        - ``posdef`` is the Hermitian positive-semidefinite factor (n x n if m >= n,
            or m x m if m < n) if compute_Hermitian is True.

    Raises:
        ValueError: If the input `a` is not 2-D or if an invalid side or method is provided.
        NotImplementedError: If the combination of matrix shape and `side` is not supported by QDWH.

    Examples:

        >>> a = torch.tensor([[1., 2., 3.],
        ...                    [5., 4., 2.],
        ...                    [3., 2., 1.]])
        >>> U, H = polar(a, compute_hermitian=True)
        >>> torch.allclose(U.T @ U, torch.eye(U.shape[1]))
        True
        >>> a_reconstructed = U @ H
        >>> torch.allclose(a, a_reconstructed)
        True
    """
    # Convert input to tensor.
    arr = torch.as_tensor(a)
    if arr.ndim != 2:
        raise ValueError("The input `a` must be a 2-D array.")

    m, n = arr.shape
    max_iterations = max_iterations if max_iterations is not None else 5

    if method == "qdwh":
        from qdwh import qdwh
        # For QDWH, we support one of two cases.
        if m >= n:
            # Call the QDWH routine on the original matrix.
            res = qdwh(arr, is_hermitian=False, compute_hermitian=compute_hermitian,
                    max_iterations=max_iterations,
                    eps=eps)
            unitary = res[0]
            if compute_hermitian:
                posdef = res[1]
        else:
            # For a left polar decomposition when m < n, work with the conjugate-transpose.
            arr_t = arr.transpose(0, 1).conj()
            res = qdwh(arr_t, is_hermitian=False, compute_hermitian=compute_hermitian,
                    max_iterations=max_iterations,
                    eps=eps)
            unitary = res[0]
            # Revert the transformation.
            unitary = unitary.transpose(0, 1).conj()
            if compute_hermitian:
                posdef = res[1]
                posdef = posdef.transpose(0, 1).conj()
    elif method == "zolo-pd":
        from zolopd import zolopd
        res = zolopd(arr, compute_hermitian=compute_hermitian)
        unitary = res[0]
        if compute_hermitian:
            posdef = res[1]
    elif method == "ns":
        from newton_schulz import zeropower_via_newtonschulz5
        res = zeropower_via_newtonschulz5(arr, compute_hermitian=compute_hermitian, max_iterations=max_iterations, a=ns_coeffs[0], b=ns_coeffs[1], c=ns_coeffs[2])
        if compute_hermitian:
            unitary, posdef = res
        else:
            unitary = res
    elif method == "precond_ns":
        from newton_schulz import precond_newtonschulz
        res = precond_newtonschulz(arr, compute_hermitian=compute_hermitian)
        unitary = res[0]
        if compute_hermitian:
            posdef = res[1]
    elif method == "polar_express":
        from polar_express import PolarExpress
        res = PolarExpress(arr, compute_hermitian=compute_hermitian, max_iterations=max_iterations)
        unitary = res[0]
        if compute_hermitian:
            posdef = res[1]
    else:
        raise ValueError(f"Unknown polar decomposition method {method}.")
    
    return unitary, posdef if compute_hermitian else unitary


class PolarGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, weight_decay=0., momentum=0.95, polar_first=False, method='qdwh', inner_steps=5, a=3.4445, b=-4.7750, c=2.031):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, polar_first=polar_first, method=method, inner_steps=inner_steps, a=a, b=b, c=c)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                orig_dtype = p.dtype
                g = p.grad.data.float() # upcast to float32
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(g)
                m = state['momentum']
                if group['polar_first']:
                    U = polar(g, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0]
                    nuc_norm = torch.sum(g.type_as(U) * U)
                    m.lerp_(U, 1 - group["momentum"])
                    g = nuc_norm * m
                else:
                    m.lerp_(g, 1 - group["momentum"])
                    U = polar(m, method=group['method'], max_iterations=group["inner_steps"], ns_coeffs=(group['a'], group['b'], group['c']))[0]
                    nuc_norm = torch.sum(m.type_as(U) * U)
                    g = nuc_norm * U
                g = g.to(orig_dtype) # Casted back to bfloat16
                p.data.mul_(1 - group['lr'] * group['weight_decay']).add_(g, alpha=-group['lr'])
        return loss


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            self.tokens = torch.load(f"{self.dataset_name}.bin")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            # import pdb; pdb.set_trace()
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def get_model_and_dataloader(model_name, dataset_name, hidden_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters())} (goal: 540,865,536")
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader

def classify_params(model):
    hidden_2d, embed_param, head_param, scalar_vec = [], [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Embedding weight matrix
        if name == "model.embed_tokens.weight":
            embed_param.append(param)
        # lm_head weight matrix
        elif name == "lm_head.weight":
            head_param.append(param)
        # 2d weight matrices in transformer hidden layers
        elif param.ndim == 2 and "layers." in name:
            hidden_2d.append(param)
        # remaining (norms, biases, etc)
        else:
            scalar_vec.append(param)

    print(f"  hidden 2D matrices : {len(hidden_2d)} tensors "
          f"({sum(p.numel() for p in hidden_2d):,} params)")
    print(f"  embed              : {sum(p.numel() for p in embed_param):,} params")
    print(f"  head               : {sum(p.numel() for p in head_param):,} params")
    print(f"  scalar/vector      : {len(scalar_vec)} tensors "
          f"({sum(p.numel() for p in scalar_vec):,} params)")
    return hidden_2d, embed_param, head_param, scalar_vec

def get_linear_decay_lr_fn(total_steps, decay_ratio=0.4):
    decay_start = int(total_steps * (1 - decay_ratio))
    def lr_fn(step):
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (total_steps - decay_start)
        return max(0.0, 1.0 - progress)
    return lr_fn


@torch.no_grad()
def gradient_diagnostics(param):
    g = param.grad.float()  # upcast for numerical accuracy
    svs = torch.linalg.svdvals(g)
    sigma_max = svs[0].item()
    sigma_min = svs[-1].item()
    cond = sigma_max / (sigma_min + 1e-12)
    nuc  = svs.sum().item()
    return cond, nuc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=1024)
    args = parser.parse_args()
    logger.add(f"logs/train_{args.model}_{args.optimizer}_lr{args.lr}.log")

    model, train_loader = get_model_and_dataloader(
        args.model, args.dataset, args.hidden_size
    )

    hidden_2d, embed_param, head_param, scalar_vec = classify_params(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dtype=torch.bfloat16)
    model.to(device)

    model.train()
    epoch = 1
    total_steps = len(train_loader)

    if args.optimizer == 'adamw':
        # AdamW for everything: lr=0.001, betas=(0.9,0.95), wd=0.1
        optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(train_loader) * epoch,
            num_cycles=0.5,
        )

    elif args.optimizer == 'muon_adamw':
        # Muon for hidden 2D weights (lr=0.001, β=0.95, ns_steps=5)
        # AdamW for embed + head + scalars (lr=0.001, betas=(0.9,0.95), no wd)
        adamw_params = embed_param + head_param + scalar_vec
        optimizer = Muon(
            muon_params=hidden_2d,
            wd=0,
            lr=args.lr, momentum=0.95, nesterov=True, ns_steps=5,
            adamw_params=adamw_params,
            adamw_betas=(0.9, 0.95), 
            adamw_eps=1e-8,
        )
        lr_fn = get_linear_decay_lr_fn(total_steps, decay_ratio=0.4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    elif args.optimizer == 'muon_polarsgdm':
        # Muon for hidden layers (lr=0.001, β=0.95, ns_steps=5)
        # PolarSGDM for embed + head (lr=0.001, β=0.5, qdwh, inner_steps=5)
        # AdamW for scalars/vectors (lr=0.001, betas=(0.9,0.95), no wd)
        optimizer_muon = Muon(
            muon_params=hidden_2d,
            lr=args.lr, wd=0, momentum=0.95, nesterov=True, ns_steps=5,
        )
        optimizer_polar = PolarGrad(
            embed_param + head_param,
            lr=args.lr, momentum=0.5, method='qdwh', inner_steps=5
        )
        optimizer_scalar = torch.optim.AdamW(
            scalar_vec, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
        )
        lr_fn = get_linear_decay_lr_fn(total_steps, decay_ratio=0.4)
        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(optimizer_muon, lr_lambda=lr_fn),
            torch.optim.lr_scheduler.LambdaLR(optimizer_polar, lr_lambda=lr_fn),
            torch.optim.lr_scheduler.LambdaLR(optimizer_scalar, lr_lambda=lr_fn),
        ]

    embed_p = embed_param[0]
    head_p = head_param[0]

    losses, cond_embed, cond_head, nuc_embed, nuc_head, steps_log = \
        [], [], [], [], [], []

    for epoch in range(epoch):
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            input_ids = batch

            if args.optimizer == 'muon_polarsgdm':
                optimizer_muon.zero_grad()
                optimizer_polar.zero_grad()
                optimizer_scalar.zero_grad()
            else:
                optimizer.zero_grad()

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()

            if args.optimizer == 'muon_polarsgdm':
                optimizer_muon.step()
                optimizer_polar.step()
                optimizer_scalar.step()
                for sched in schedulers:
                    sched.step()
            else:
                optimizer.step()
                scheduler.step()

            if step % 50 == 0:
                cond_e, nuc_e = gradient_diagnostics(embed_p)
                cond_h, nuc_h = gradient_diagnostics(head_p)
                losses.append(loss.item())
                cond_embed.append(cond_e)
                cond_head.append(cond_h)
                nuc_embed.append(nuc_e)
                nuc_head.append(nuc_h)
                steps_log.append(step)
                current_lr = (optimizer_muon if args.optimizer == 'muon_polarsgdm' 
                            else optimizer).param_groups[0]['lr']
                logger.info(
                    f"step={step} loss={loss.item():.4f} "
                    f"kappa_embed={cond_e:.2f} kappa_head={cond_h:.2f} "
                    f"nuc_embed={nuc_e:.4f} nuc_head={nuc_h:.4f} "
                    f"lr={current_lr:.6f}"
                )
    # Saving to json 
    os.makedirs("results", exist_ok=True)
    results = {
        "optimizer": args.optimizer,
        "lr": args.lr,
        "steps": steps_log,
        "loss": losses,
        "cond_embed": cond_embed,
        "cond_head": cond_head,
        "nuc_embed": nuc_embed,
        "nuc_head": nuc_head,
    }
    out_path = f"results/{args.model}_{args.optimizer}_lr{args.lr}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)
    logger.info(f"Saved results to {out_path}")
