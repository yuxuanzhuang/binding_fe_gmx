# find run folders in us folder
for folder in $(find us -type d -name "run*"); do
    cd $folder
    echo "Running in $folder"
    mkdir eq
    gmx grompp -f pulling_eq.mdp -o eq/pulling_eq.tpr -p ../../topol.top -c start.pdb -r start.pdb -maxwarn 2
    cd eq
    gmx mdrun -v -deffnm pulling_eq
    cd ../../../
done