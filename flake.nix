{
  description = "CUDA-enabled Jupyter notebooks using uv on NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaVersion = "12";
          };
        };

        nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          name = "cuda-uv-jupyter-env";

          buildInputs = with pkgs; [

            # Toolchain / basics
            gcc
            gcc.cc.lib
            stdenv.cc
            binutils
            zlib
            ncurses

            # GL/X libs (useful for some plotting / GUI stuff)
            libGLU
            libGL
            mesa
            xorg.libXi
            xorg.libXmu
            freeglut
            xorg.libXext
            xorg.libX11
            xorg.libXv
            xorg.libXrandr
            glib

            # Python + uv
            python
            uv
          ];

          shellHook = ''
            Next steps (inside this shell):

              1) uv sync

              2) Start Jupyter (using your project script):
                   uv run jupyter-lab -- --ip=127.0.0.1 --no-browser

              3) Then test CUDA in a notebook:
                   import torch
                   torch.cuda.is_available()
                   torch.cuda.get_device_name(0)

            EOF
          '';
        };
      }
    );
}
