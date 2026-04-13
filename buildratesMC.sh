#!/bin/bash
# buildratesMC.sh — Generate RatesMC input files for all (a,p) reactions.
#
# Reads runtime parameters from tesseract.in (set TESSERACT_IN env var to
# override the default path).  Only E_min_mev, E_max_mev, samples, runs,
# mean_i, mean_o, input_dir, and output_dir are consumed here; all other
# parameters are handled by build_ratesmc_input.py / tesseract.py.

TESSERACT_IN="${TESSERACT_IN:-tesseract.in}"

# ---------------------------------------
# Read a key from tesseract.in
# Usage: parse_param KEY DEFAULT
# Searches all sections; first match wins.
# ---------------------------------------
parse_param() {
    local key="$1"
    local default="$2"
    local val
    val=$(grep -E "^\s*${key}\s*=" "$TESSERACT_IN" 2>/dev/null \
          | head -1 \
          | sed 's/[^=]*=[ \t]*//' \
          | sed 's/[ \t]*#.*//' \
          | sed 's/^[ \t]*//; s/[ \t]*$//')
    echo "${val:-$default}"
}

if [ ! -f "$TESSERACT_IN" ]; then
    echo "❌  tesseract.in not found at: $TESSERACT_IN"
    echo "    Set TESSERACT_IN=<path> to specify a different location."
    exit 1
fi

# ---------------------------------------
# Read parameters from tesseract.in
# ---------------------------------------
Input_dir=$(parse_param  input_dir   "input/")
Output_dir=$(parse_param output_dir  "outputs/")
E_min=$(parse_param      E_min_mev   "0.1")
E_max=$(parse_param      E_max_mev   "10.0")
samples=$(parse_param    samples     "5000")
runs=$(parse_param       runs        "1")
mean_i=$(parse_param     mean_i      "0.010")
mean_o=$(parse_param     mean_o      "0.0045")

echo "Reading parameters from: $TESSERACT_IN"
echo "  input_dir  = $Input_dir"
echo "  output_dir = $Output_dir"
echo "  E_min      = $E_min MeV"
echo "  E_max      = $E_max MeV"
echo "  samples    = $samples"
echo "  runs       = $runs"
echo "  mean_i     = $mean_i"
echo "  mean_o     = $mean_o"
echo "------------------------------------------------------------"

# ---------------------------------------
# Select all (a,p) reaction templates
# ---------------------------------------
input_files=(${Input_dir}*\(a\,p\)*.txt)

if [ ${#input_files[@]} -eq 0 ]; then
    echo "❌  No (a,p) template files found in $Input_dir"
    exit 1
fi

# ---------------------------------------
# Loop over all reaction files
# ---------------------------------------
for ((i=0; i<${#input_files[@]}; i++)); do

    file="${input_files[$i]}"
    filename=$(basename -- "$file")
    prefix="${filename%.txt}"
    echo "$prefix"

    # ---------------------------------------
    # Extract A1 (target mass number) from filename
    # Example: 22Mg(a,p)25Al.txt → A1=22
    # ---------------------------------------
    A1=$(echo "$filename" | grep -o '^[0-9]\+')

    if [ -z "$A1" ]; then
        echo "❌  Could not extract A1 from $filename"
        continue
    fi

    echo "Processing $filename (A1=$A1)"

    # ---------------------------------------
    # Compute gamma values using Python
    # (same Wigner-limit calculation as tesseract.py::compute_gamma)
    # ---------------------------------------
    read gamma_i_mean_ev gamma_o_mean_ev m_target m_alpha <<< $(python - << EOF
from nucres.physics import HBAR, MASS_PROTON, reduced_mass

A1 = $A1
A2_alpha  = 4
A2_proton = 1

m_alpha  = A2_alpha  * MASS_PROTON
m_target = A1        * MASS_PROTON

mu_i   = reduced_mass(m_target, m_alpha)
R_sq_i = (1.25e-15)**2 * (A1**(1/3) + A2_alpha**(1/3))**2
wig_i  = 3*HBAR**2 / (2*mu_i*R_sq_i) * 6.242e18
gamma_i_mean_ev = wig_i * $mean_i

mu_o   = reduced_mass(m_target, A2_proton * MASS_PROTON)
R_sq_o = (1.25e-15)**2 * (A1**(1/3) + A2_proton**(1/3))**2
wig_o  = 3*HBAR**2 / (2*mu_o*R_sq_o) * 6.242e18
gamma_o_mean_ev = wig_o * $mean_o

print(gamma_i_mean_ev, gamma_o_mean_ev, m_target, m_alpha)
EOF
)

    echo "  gamma_i_mean_ev = $gamma_i_mean_ev"
    echo "  gamma_o_mean_ev = $gamma_o_mean_ev"
    echo "  m_target        = $m_target"
    echo "  m_alpha         = $m_alpha"

    # ---------------------------------------
    # Loop over runs
    # ---------------------------------------
    for ((j=0; j<$runs; j++)); do

        run_dir="${Output_dir}${prefix}/RUN_$j"
        out_in="${run_dir}/${prefix}.in"

        # Skip this run if output already exists (modular restart)
        if [ -f "$out_in" ]; then
            echo "  → RUN_$j already exists ($out_in) — skipping."
            continue
        fi

        mkdir -p "$run_dir"

        echo "  → Run $j (E_min=$E_min, E_max=$E_max)"

        python build_ratesmc_input.py \
            --template        "$file" \
            --output-dir      "${Output_dir}" \
            --E-min-mev       "$E_min" \
            --E-max-mev       "$E_max" \
            --m1              "$m_target" \
            --m2              "$m_alpha" \
            --n-density-points "$samples" \
            --Gamma-i-mean-eV "$gamma_i_mean_ev" \
            --Gamma-o-mean-eV "$gamma_o_mean_ev"

        mv "${Output_dir}${prefix}/${prefix}.in" "$run_dir/"

    done

    echo "Generated RatesMC input for $filename"
    echo "------------------------------------------------------------"

done
