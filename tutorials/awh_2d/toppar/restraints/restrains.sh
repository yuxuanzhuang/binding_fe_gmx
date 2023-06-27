declare -a arr=('a' 'b' 'c' 'd' 'e' 'f' 'g' 'h' 'i' 'j')
for index in "${arr[@]}"
do
gmx select -f ../../em.pdb -s ../../em.pdb -ofpdb pro$index.pdb -pdbatoms selected<<EOF
chain ${index^^}
EOF
gmx genrestr -f pro$index.pdb -o posre_pro$index.itp<<EOF
2
EOF
gmx genrestr -f pro$index.pdb -o posre_backbone_pro$index.itp<<EOF
4
EOF
gmx genrestr -f pro$index.pdb -o posre_ca_pro$index.itp<<EOF
3
EOF
done
