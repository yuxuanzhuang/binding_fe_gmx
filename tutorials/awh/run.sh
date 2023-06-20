## Step 1
# Move two molecules close to each other and merge to start.pdb

## Step 2
## create a suitable size box
gmx editconf -f start.pdb -d 1.5 -o box.pdb

## Step 3.1
## solvate the box
gmx solvate -cp box.pdb -cs -o solvated.pdb -p topol.top

## Step 3.2
## change name of SOL to TIP3 in topol.top

## Step 3.3
## add ions and make sure the box is charge neutral
touch ion.mdp
gmx grompp -f ion.mdp -o ion.tpr -p topol.top -c solvated.pdb -r solvated.pdb -maxwarn 2
echo 4 | gmx genion -s ion.tpr -np 4 -nn 5 -o ion.pdb -pname SOD -nname CLA -p topol.top

## Step 4
## em
gmx grompp -f mdp/em.mdp -o em.tpr -p topol.top -c ion.pdb -r ion.pdb -maxwarn 2
gmx mdrun -v -deffnm em

## Step 5
## NVT eq
gmx grompp -f mdp/nvt.mdp -o nvt.tpr -p topol.top -c em.gro -r em.gro -maxwarn 2
gmx mdrun -v -deffnm nvt

## Step 6.1
## NPT eq
gmx grompp -f mdp/npt.mdp -o npt.tpr -p topol.top -c nvt.gro -r nvt.gro -maxwarn 2
gmx mdrun -v -deffnm npt

## Step 6.2
## convert gro to pdb file
echo 0 | gmx trjconv -f npt.gro -s npt.tpr -o eq.pdb

## Step 7
## generate AWH walkers
mkdir AWH
mkdir AWH/rep{1..4}
gmx select -f eq.pdb -s eq.pdb -on index.ndx
## select 0,2,3
gmx grompp -f mdp/awh.mdp -o AWH/awh.tpr -c eq.pdb -r eq.pdb -p topol.top -maxwarn 2 -n index.ndx
for rep in {1..4};
    do cp AWH/awh.tpr AWH/rep$rep;
done

sbatch jobscript.sh