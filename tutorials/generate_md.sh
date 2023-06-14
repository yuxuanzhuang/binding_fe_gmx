# find run folders in us folder
for folder in $(find us -type d -name "run*"); do
    cd $folder
    echo "Running in $folder"
    gmx grompp -f pulling_md.mdp -o pulling_md.tpr -p ../../topol.top -c eq/pulling_eq.gro -r eq/pulling_eq.gro -maxwarn 2
#    gmx mdrun -v -deffnm pulling_eq
    cd ../../
done

# get number of pulling_md.tpr files
num_tpr=$(ls us/run*/pulling_md.tpr | wc -l)
echo "Number of pulling_md.tpr files: $num_tpr"