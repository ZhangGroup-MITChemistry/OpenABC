#!/bin/bash
#SBATCH --job-name=mpipi
#SBATCH --output=slurm-output/mpipi-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10GB
#SBATCH --time=12:00:00

lmp_src=/home/gridsan/sliu/my-tools/lammps-mpipi/lammps-patch_10Mar2021/src

${lmp_src}/lmp_serial -in lammps_Mpipi_RNA.in

echo "job done"

