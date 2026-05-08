{
  description = "PolarGrad experiments — CPU-only Python 3.12 environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python312;

      pythonEnv = python.withPackages (ps: with ps; [
        pytorch
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

      # Targeted TeX Live combo for matplotlib usetex + scienceplots styles.
      # Keeps the closure ~1 GB instead of the ~7 GB scheme-full would pull.
      tex = pkgs.texlive.combine {
        inherit (pkgs.texlive)
          scheme-basic
          type1cm        # the package missing in your error
          cm-super       # Type 1 versions of CM fonts (matplotlib needs these)
          underscore
          dvipng         # matplotlib's usetex pipeline
          amsmath
          amsfonts
          siunitx        # scienceplots
          mathtools      # scienceplots
          physics        # scienceplots
          booktabs
          helvetic
          palatino
          times;
      };
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pythonEnv
          tex
          pkgs.ghostscript  # matplotlib uses gs for some usetex paths
          pkgs.git
          pkgs.gcc
        ];

        shellHook = ''
          echo "── PolarGrad dev shell (CPU-only PyTorch) ─────────────────"
          echo "Python : $(python --version)"
          python - <<'PY'
import torch
print(f"Torch  : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
PY
          echo "LaTeX  : $(latex --version | head -1)"
          echo
          echo "Note: matrix experiments run fine on CPU. Qwen pretraining"
          echo "      is impractical without an NVIDIA/Ampere+ GPU."
          echo "──────────────────────────────────────────────────────────"

          export PYTHONPATH="$PWD/polargrad:$PYTHONPATH"
        '';
      };
    };
}
