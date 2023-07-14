#!/bin/bash

# Submit to the tcb partition
#SBATCH -p lindahl4

# The name of the job in the queue
#SBATCH -J epj_new_modified_2
# wall-clock time given to this job
#SBATCH -t 24:00:00

# Number of nodes and number of MPI processes per node
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G 8

# Output file names for stdout and stderr
#SBATCH -e job-%j.err -o job-%j.out

# Receive e-mails when your job starts and ends

#SBATCH -d singleton
#export GMX_DISABLE_GPU_TIMING=1

# The actual script starts here
module unload gromacs
module switch gromacs/2023 gromacs=gmx_mpi
module switch cuda/11.8
module unload openmpi
module load openmpi
srun -n 8 gmx_mpi mdrun -deffnm awh -cpi awh -multidir rep{1..8} -awh awhinit.xvg -px awh_pullx -pf awh_pullf -maxh 23 -dhdl dhdl