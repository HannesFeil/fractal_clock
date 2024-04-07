{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system: 
      let 
        pkgs = import nixpkgs { inherit system; overlays = [ nixgl.overlay ]; };
      in with pkgs;
        {
          devShell = mkShell rec {
            buildInputs = [
              libxkbcommon
              libGL

              # WINIT_UNIX_BACKEND=wayland
              wayland          
            ];
            packages = [
              pkgs.nixgl.nixGLIntel
            ];
            LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
          };
        }
    );
}
