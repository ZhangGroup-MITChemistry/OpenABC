#!/bin/bash
set -euo pipefail

# Path 1: folder containing protein.pdb files
src_dir="/home/yumzhang/orcd/pool/work/4-idpcg/git_openabc_moff2/CL_idp_test_reweight/"


# Path 2: current directory
dst_dir="$(pwd)"

for pdb in "$src_dir"/*_ca.pdb; do
    [[ -e "$pdb" ]] || continue

    # Get protein name by removing _ca.pdb
    protein=$(basename "$pdb" _ca.pdb)

    # Make target folder: ./protein/
    mkdir -p "$dst_dir/$protein"

    # Copy path1/protein_ca.pdb to ./protein/protein.pdb
    cp "$pdb" "$dst_dir/$protein/$protein.pdb"

    echo "Copied: $pdb -> $dst_dir/$protein/$protein.pdb"
done
