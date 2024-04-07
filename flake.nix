{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system: 
      let 
        pkgs = import nixpkgs { inherit system; };
      in with pkgs;
        {
          devShell = mkShell rec {
            buildInputs = [
              libxkbcommon
              libGL
              vulkan-loader

              # WINIT_UNIX_BACKEND=wayland
              wayland          
            ];
            LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
          };
        }
    );
}
