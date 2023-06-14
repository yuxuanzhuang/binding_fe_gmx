#!/bin/bash

# Submit to the tcb partition
#SBATCH -p lindahl1,lindahl2,lindahl3,lindahl4

# The name of the job in the queue
#SBATCH -J us_epj
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
cd us
# run 4 replicas simultaneously from run_1 to run_40
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{0..3} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{4..7} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{8..11} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{12..15} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{16..19} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{20..23} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{24..27} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{28..31} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{32..35} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23
srun -n 4 gmx_mpi mdrun -deffnm pulling_md -cpi pulling_md -multidir run_{36..39} -pme gpu -bonded gpu -nb gpu -px pullx -pf pullf -maxh 23