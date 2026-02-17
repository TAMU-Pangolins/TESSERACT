#!/bin/bash

# Format:
# name Ztarget Atarget Sproj Sexit

reactions=(
"14O(a,p)17F 8 14 5115.0969 3923.0711"
"18Ne(a,p)21Na 10 18 8142.51887 5504.100064"
"26Si(a,p)29P 14 26 9343.16687 4395.376064"
"30S(a,p)33Cl 16 30 6743.95487 4663.924064"
"34Ar(a,p)37K 18 34 6105.12587 4547.273064"
"38Ca(a,p)41Sc 20 38 5470.76587 3750.964064"
)

for r in "${reactions[@]}"; do
    set -- $r
    name=$1
    Zt=$2
    At=$3
    Sproj=$4
    Sexit=$5

    cat > "$name.txt" <<EOF
$name
*************************************************** *************************************************** **********
2 ! Zproj
$Zt ! Ztarget
1 ! Zexitparticle (=0 when only 2 channels open)
4.003 ! Aproj
$At ! Atarget
1.009 ! Aexitparticle (=0 when only 2 channels open)
0.0 ! Jproj
0.0 ! Jtarget
0.5 ! Jexitparticle (=0 when only 2 channels open)
$Sproj ! projectile separation energy (keV)
$Sexit ! exit particle separation energy (=0 when only 2 channels open)
1.25 ! Radius parameter R0 (fm)
3 ! Gamma-ray channel number (=2 if ejectile is a g-ray; =3 otherwise)
EOF

done

