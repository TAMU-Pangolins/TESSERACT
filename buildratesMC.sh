#!/bin/bash

Input_dir="input/"
Output_dir="outputs/"

# -------------------------
# OPTION A: Select all files
# -------------------------
# input_files=(${Input_dir}*.txt)

# -------------------------
# OPTION B: Select specific files
# -------------------------
input_files=(
    "${Input_dir}22Mg(a,p)25Al.txt"
)

# -------------------------
# E_min and E_max arrays (may be shorter!). If length of E_min and E_max don't match with each other or with length of input_files then default E_min and E_max values will be used.
# -------------------------
E_min=(0.01)          # Example: only 1 value
E_max=(3.8)  # Example: more values than needed
bin_width=(0.1)
# -------------------------
# Default values
# -------------------------
DEFAULT_EMIN=0.01
DEFAULT_EMAX=10.0

# -------------------------
# Loop over files
# -------------------------
for ((i=0; i<${#input_files[@]}; i++)); do

    file="${input_files[$i]}"
    filename=$(basename -- "$file")
    prefix="${filename%.txt}"

    # -------------------------------------------
    # Safely assign E_min[i] or fallback to default
    # -------------------------------------------
    if [ $i -lt ${#E_min[@]} ]; then
        Emin_val="${E_min[$i]}"
    else
        Emin_val=$DEFAULT_EMIN
        echo "⚠️  WARNING: No E_min value for $filename → using default ($DEFAULT_EMIN)"
    fi

    # -------------------------------------------
    # Safely assign E_max[i] or fallback to default
    # -------------------------------------------
    if [ $i -lt ${#E_max[@]} ]; then
        Emax_val="${E_max[$i]}"
    else
        Emax_val=$DEFAULT_EMAX
        echo "⚠️  WARNING: No E_max value for $filename → using default ($DEFAULT_EMAX)"
    fi

    echo "Processing $filename (E_min=$Emin_val, E_max=$Emax_val)"

    python build_ratesmc_input.py \
        --template "$file" \
        --output-dir "$Output_dir" \
        --E-min-mev "$Emin_val" \
        --E-max-mev "$Emax_val" \
        --delta-E-mev "$bin_width"

    echo "Generated RatesMC input file for $filename in $Output_dir"
    echo "------------------------------------------------------------"
done

