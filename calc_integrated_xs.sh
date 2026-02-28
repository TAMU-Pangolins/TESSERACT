#!/bin/bash

# Energy grid parameters
EMIN=0.1
EMAX=10.2
ERES=0.001

# List of integration bin widths
dE_val="$@"

REACTIONS=(
"14O(a,p)17F"
"18Ne(a,p)21Na"
"22Mg(a,p)25Al"
"26Si(a,p)29P"
"30S(a,p)33Cl"
"34Ar(a,p)37K"
"38Ca(a,p)41Sc"
)

echo "Starting to merge parts + integrating cross sections for the following reactions: ${REACTIONS[@]}"

for RXN in "${REACTIONS[@]}"; do

    echo "-------------------------------------------------"
    echo "Processing $RXN"
    echo "-------------------------------------------------"

    python calc_xs_distributed_parallel.py --reaction "$RXN" --Emin $EMIN --Emax $EMAX --E_resol $ERES --dE $dE_val --merge_only

    if [ $? -ne 0 ]; then
        echo "Error occurred for reaction $RXN"
        exit 1

    fi

    echo "Completed $RXN"

done

echo "-----------------------------------"
echo "All reactions processed successfully"
