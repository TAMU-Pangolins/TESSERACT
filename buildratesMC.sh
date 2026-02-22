#!/bin/bash
Input_dir="input/"
Output_dir="outputs/"

# ---------------------------------------
# Select all .txt files automatically
# ---------------------------------------
input_files=(${Input_dir}*\(a\,p\)*.txt)

# ---------------------------------------
# Energy settings
# ---------------------------------------
E_min=(0.1)
E_max=(10.0)

samples=5000
runs=1

DEFAULT_EMIN=0.1
DEFAULT_EMAX=10.0

# ---------------------------------------
# Loop over all reaction files
# ---------------------------------------
for ((i=0; i<${#input_files[@]}; i++)); do

    file="${input_files[$i]}"
    filename=$(basename -- "$file")
    prefix="${filename%.txt}"
    echo "$prefix"

    # ---------------------------------------
    # Extract A1 from filename (leading digits)
    # Example: 22Mg(a,p)25Al.txt → A1=22
    # ---------------------------------------
    A1=$(echo "$filename" | grep -o '^[0-9]\+')

    if [ -z "$A1" ]; then
        echo "❌ Could not extract A1 from $filename"
        continue
    fi

    echo "Processing $filename (A1=$A1)"

    # ---------------------------------------
    # Assign Emin / Emax safely
    # ---------------------------------------
    if [ $i -lt ${#E_min[@]} ]; then
        Emin_val="${E_min[$i]}"
    else
        Emin_val=$DEFAULT_EMIN
    fi

    if [ $i -lt ${#E_max[@]} ]; then
        Emax_val="${E_max[$i]}"
    else
        Emax_val=$DEFAULT_EMAX
    fi

    # ---------------------------------------
    # Compute gamma values using Python
    # ---------------------------------------
    read gamma_i_mean_ev gamma_o_mean_ev m_target m_alpha <<< $(python - << EOF
from nucres.physics import HBAR, MASS_PROTON, reduced_mass

A1 = $A1
A2_alpha = 4
A2_proton = 1

m_alpha = A2_alpha * MASS_PROTON
m_target = A1 * MASS_PROTON

# Incoming (alpha channel)
mu_i = reduced_mass(m_target, m_alpha)
R_sq_i = (1.25e-15)**2 * (A1**(1/3) + A2_alpha**(1/3))**2
wig_i = 3*HBAR**2 / (2*mu_i*R_sq_i) * 6.242e18
gamma_i_mean_ev = wig_i * 0.010

# Outgoing (proton channel)
mu_o = reduced_mass(A1*MASS_PROTON, A2_proton*MASS_PROTON)
R_sq_o = (1.25e-15)**2 * (A1**(1/3) + A2_proton**(1/3))**2
wig_o = 3*HBAR**2 / (2*mu_o*R_sq_o) * 6.242e18
gamma_o_mean_ev = wig_o * 0.0045

print(gamma_i_mean_ev, gamma_o_mean_ev, m_target, m_alpha)
EOF
)

    echo "gamma_i_mean_ev = $gamma_i_mean_ev"
    echo "gamma_o_mean_ev = $gamma_o_mean_ev"
    echo "m1 = $m_target"
    echo "m2 = $m_alpha"

    # ---------------------------------------
    # Loop over runs
    # ---------------------------------------
    for ((j=0; j<$runs; j++)); do

        run_dir="${Output_dir}${prefix}/RUN_$j"
        mkdir -p "$run_dir"

        echo "  → Run $j (E_min=$Emin_val, E_max=$Emax_val)"

        python build_ratesmc_input.py \
            --template "$file" \
            --output-dir "${Output_dir}" \
            --E-min-mev "$Emin_val" \
            --E-max-mev "$Emax_val" \
            --m1 "$m_target" \
            --m2 "$m_alpha" \
            --n-density-points "$samples" \
            --Gamma-i-mean-eV "$gamma_i_mean_ev" \
            --Gamma-o-mean-eV "$gamma_o_mean_ev" \
            --seed 42

        mv "${Output_dir}${prefix}/${prefix}.in" "$run_dir/"

    done

    echo "Generated RatesMC input for $filename"
    echo "------------------------------------------------------------"

done
