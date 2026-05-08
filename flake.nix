{
  description = "PolarGrad experiments — CPU-only Python 3.12 environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;

      # All Python deps from pixi.toml, minus CUDA-only pieces (pytorch-cuda, triton).
      # `pytorch` from nixpkgs is the CPU build by default — works on AMD/CPU.
      pythonEnv = python.withPackages (ps: with ps; [
        torch
        transformers
        datasets
        matplotlib
        numpy
        tqdm
        scienceplots
        tiktoken
        loguru
        fire
        setuptools
      ]);
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          pkgs.git           # needed for `git submodule update --init --recursive`
          pkgs.gcc           # some torch ops jit-compile small C kernels
        ];

        shellHook = ''
          echo "── PolarGrad dev shell (CPU-only PyTorch) ─────────────────"
          echo "Python : $(python --version)"
          python - <<'PY'
import torch
print(f"Torch  : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
PY
          echo
          echo "Note: matrix experiments (NMF, multi-response, multinomial)"
          echo "      run fine on CPU. Qwen2.5 pretraining is impractical"
          echo "      without an NVIDIA/Ampere+ GPU."
          echo "──────────────────────────────────────────────────────────"

          # Make the vendored polargrad/ submodule importable without installing it.
          export PYTHONPATH="$PWD/polargrad:$PYTHONPATH"
        '';
      };
    };
}
