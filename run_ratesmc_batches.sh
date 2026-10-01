#!/usr/bin/env bash
# Helper to build a fresh RatesMC.in for each run and execute RatesMC in
# per-reaction, per-run directories:
#   <OUTPUT_ROOT>/<reaction_name>/Run_<NNNN>/
# Usage:
#   RUNS=100 RATESMC_BIN=/path/to/RatesMC ./run_ratesmc_batches.sh input/*.txt
#
# Env vars:
#   RATESMC_BIN   Path to the compiled RatesMC executable (required).
#   RUNS          Number of Runs per reaction (default: 10).
#   OUTPUT_ROOT   Base directory for outputs (default: outputs).
#   MASS_DIR      Directory holding mass_1.mas20 / nubase_3.mas20 (default:
#                 beside RATESMC_BIN, or one level up for an upstream build/).
#
# RatesMC writes its own RatesMC.log; console output goes to RatesMC.stdout.
# A run counts as failed when RatesMC.out has no rate rows (the exit code is
# not used: the RatesMC 2.2+ rewrite returns 1 even on success). The script
# exits non-zero if any run failed.

set -euo pipefail

resolve_ratesmc_bin() {
    if [ -n "${RATESMC_BIN:-}" ]; then
        echo "$RATESMC_BIN"
        return
    fi
    # Try to find RatesMC on PATH (Windows and Unix filenames).
    if command -v RatesMC >/dev/null 2>&1; then
        command -v RatesMC
        return
    fi
    if command -v ratesmc >/dev/null 2>&1; then
        command -v ratesmc
        return
    fi
    echo ""  # not found
}

RATESMC_BIN="$(resolve_ratesmc_bin)"

if [ -z "$RATESMC_BIN" ]; then
    echo "RatesMC executable not found. Set RATESMC_BIN or put RatesMC on your PATH." >&2
    exit 1
fi

if [ ! -x "$RATESMC_BIN" ]; then
    echo "RATESMC_BIN is not executable: $RATESMC_BIN" >&2
    exit 1
fi

resolve_python() {
    if [ -n "${PYTHON:-}" ] && command -v "$PYTHON" >/dev/null 2>&1; then
        echo "$PYTHON"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return
    fi
    echo ""
}

PYTHON_BIN="$(resolve_python)"

if [ -z "$PYTHON_BIN" ]; then
    echo "Python interpreter not found. Install Python 3 or set PYTHON to your interpreter path." >&2
    exit 1
fi

RUNS="${RUNS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

# Allow passing templates on the command line; otherwise default to input/*.txt.
if [ "$#" -gt 0 ]; then
    templates=("$@")
else
    templates=(input/*.txt)
fi

if [ ${#templates[@]} -eq 0 ]; then
    echo "No template files found. Pass templates explicitly or ensure input/*.txt exists." >&2
    exit 1
fi

link_or_copy_ratesmc() {
    local dest_dir="$1"
    if ln -sf "$RATESMC_BIN" "$dest_dir/RatesMC" 2>/dev/null; then
        return 0
    fi
    cp "$RATESMC_BIN" "$dest_dir/RatesMC"
}

find_mass_dir() {
    local bin_dir
    bin_dir="$(cd "$(dirname "$(readlink -f "$RATESMC_BIN")")" && pwd)"
    for dir in "${MASS_DIR:-}" "$bin_dir" "$(dirname "$bin_dir")"; do
        if [ -n "$dir" ] && [ -f "$dir/mass_1.mas20" ] && [ -f "$dir/nubase_3.mas20" ]; then
            echo "$dir"
            return
        fi
    done
    echo ""
}

SUPPORT_DIR="$(find_mass_dir)"
if [ -z "$SUPPORT_DIR" ]; then
    echo "WARNING: mass_1.mas20 / nubase_3.mas20 not found (set MASS_DIR); RatesMC 2.2+ needs them." >&2
fi

copy_support_files() {
    local dest_dir="$1"
    [ -n "$SUPPORT_DIR" ] || return 0
    for fname in mass_1.mas20 nubase_3.mas20; do
        if ln -sf "${SUPPORT_DIR}/${fname}" "${dest_dir}/${fname}" 2>/dev/null; then
            continue
        fi
        cp "${SUPPORT_DIR}/${fname}" "${dest_dir}/"
    done
}

# Number of temperature rows (numeric first column, >= 4 columns) in a RatesMC .out.
rate_rows() {
    [ -f "$1" ] || { echo 0; return; }
    awk 'NF >= 4 && $1 ~ /^[0-9.]+([eE][-+]?[0-9]+)?$/ { n++ } END { print n + 0 }' "$1"
}

n_failed=0

for template in "${templates[@]}"; do
    if [ ! -f "$template" ]; then
        echo "Skipping missing template: $template" >&2
        continue
    fi

    reaction_name="$(basename "$template" .txt)"
    reaction_dir="${OUTPUT_ROOT}/${reaction_name}"
    mkdir -p "$reaction_dir"

    for run_idx in $(seq -w 1 "$RUNS"); do
        run_dir="${reaction_dir}/Run_${run_idx}"
        mkdir -p "$run_dir"
        link_or_copy_ratesmc "$run_dir"
        copy_support_files "$run_dir"

        input_path="${run_dir}/RatesMC.in"
        echo "Generating input for ${reaction_name} Run ${run_idx} -> ${input_path}"
        "$PYTHON_BIN" build_ratesmc_input.py \
            --template "$template" \
            --output "$input_path" \
            --spacing-model "${SPACING_MODEL:-poisson}"

        echo "Running RatesMC for ${reaction_name} Run ${run_idx}"
        rm -f "$run_dir/RatesMC.out"   # never mistake a stale table for this run's
        status=0
        (cd "$run_dir" && ./RatesMC > RatesMC.stdout 2>&1) || status=$?
        if [ "$(rate_rows "$run_dir/RatesMC.out")" -eq 0 ]; then
            echo "ERROR: RatesMC produced no rates for ${reaction_name} Run ${run_idx} (exit ${status}); see ${run_dir}/RatesMC.stdout" >&2
            n_failed=$((n_failed + 1))
        elif grep -qE "ERROR|WARNING|FATAL" "$run_dir/RatesMC.stdout"; then
            echo "RatesMC reported warnings/errors for ${reaction_name} Run ${run_idx}; see ${run_dir}/RatesMC.stdout" >&2
        fi
    done
done

if [ "$n_failed" -gt 0 ]; then
    echo "${n_failed} RatesMC run(s) failed." >&2
    exit 1
fi
