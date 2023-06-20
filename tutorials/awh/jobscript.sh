#!/bin/bash

# Submit to the tcb partition
#SBATCH -p lindahl1,lindahl2,lindahl3,lindahl4

# The name of the job in the queue
#SBATCH -J awh_epj
# wall-clock time given to this job
#SBATCH -t 24:00:00

# Number of nodes and number of MPI processes per node
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -G 4
#SBATCH --mem=10G

# Output file names for stdout and stderr
#SBATCH -e job-%j.err -o job-%j.out
#SBATCH -d singleton

# The actual script starts here
module unload gromacs
module switch gromacs/2023 gromacs=gmx_mpi
module switch cuda/11.8
module unload openmpi
module load openmpi

# Run the simulation
cd AWH
srun -n 4 gmx_mpi mdrun -deffnm awh -cpi awh -multidir rep{1..4} -pme gpu -bonded gpu -nb gpu -px awh_pullx -pf awh_pullf -maxh 23