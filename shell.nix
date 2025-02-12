{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSEnv {
  name = "simple-cargo-env";
  targetPkgs = pkgs: (with pkgs; [
    clang
    cargo
    ninja

    file
    opencv
    llvmPackages.libclang.lib
  ]);
  runScript = "cargo run -- --input-path=/mnt/share/Pictures/Wallpapers/ --output-path=/mnt/share/Pictures/WallpapersOutput --upscaler-path=./";
}).env
