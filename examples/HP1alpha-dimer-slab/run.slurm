#!/bin/bash
#SBATCH --job-name=example
#SBATCH --output=slurm-output/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --exclude=d-14-7-1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=10GB
#SBATCH --time=4-00:00:00

module load cuda/10.0

# build system
python build_system.py

# compress the system to condensate
python compress.py

# run slab simulation
T=260
output_dcd=slab_${T}K.dcd
python run_slab.py --temperature ${T} --box_a 25 --box_b 25 --box_c 400 --output_dcd ${output_dcd} --output_interval 20000 --steps 20000000

# align slab trajectory so that the largest cluster is at the center of the box
# the aligned center of mass (COM) coordinates are saved in ${aligned_COM_traj_npy}
aligned_output_dcd=aligned_slab_${T}K.dcd
aligned_COM_traj_npy=aligned_COM_traj.npy
python align_slab_traj.py ${output_dcd} ${aligned_output_dcd} ${aligned_COM_traj_npy}

# compute density profile
# use the second half of the trajectory, which is of frame index 500-999
# use regime -10<=z<=10 to compute density
start_frame_id=500
end_frame_id=999
boundary=10
output_csv=slab_density_${T}K.csv
python compute_density.py ${aligned_COM_traj_npy} ${start_frame_id} ${end_frame_id} ${boundary} ${output_csv}

# draw density profile
mkdir -p pictures
output_plot=pictures/slab_density_${T}K.pdf
python draw_density.py ${output_csv} ${output_plot}

echo "job done"


