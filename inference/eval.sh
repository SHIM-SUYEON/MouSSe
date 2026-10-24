#!/usr/bin/env bash

set -euo pipefail

# Paths 
SCRIPT="$(dirname "$0")/eval.py"
CKPT="/data/MOUSSE/mousse-checkpoint.pt"
OUT_BASE="results/mousse_uq"
LOG_DIR="logs/mousse_uq_1300"

# Config 
GPU="0"
CAP_MAIN=1300  # full test set for final reported numbers
CAP_SWEEP=1300 # adaptive lambda-scale sweep

mkdir -p "$LOG_DIR"

# Helpers 
_run() {
    local log="$1"; shift
    echo "[RUN] $* → $log"
    python3 -u "$SCRIPT" "$@" > "$log" 2>&1
}

run_adaptive() {
    local scale="$1" cap="${2:-$CAP_SWEEP}"
    local tag; tag=$(echo "$scale" | tr '.' '_')
    local log="$LOG_DIR/adaptive_scale_${tag}.log"
    local out="$OUT_BASE/adaptive_scale_${tag}"
    _run "$log" --ckpt "$CKPT" \
        --lambda-scale "$scale" --gpu "$GPU" --cap "$cap" --out "$out"
}

run_main() {
    echo ""
    echo "======================================================================"
    echo " MAIN  (CAP=$CAP_MAIN)"
    echo "======================================================================"

    # Backbone baseline with fixed alpha 1.5
    LOG_BB="$LOG_DIR/main_backbone.log"
    _run "$LOG_BB" \
        --ckpt "$CKPT" --ablation no_uq \
        --gpu "$GPU" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_backbone"

    # Adaptive UQ guidance at lambda scale 1.0
    LOG_AD="$LOG_DIR/main_adaptive_1_0.log"
    _run "$LOG_AD" \
        --ckpt "$CKPT" --lambda-scale 1.0 \
        --gpu "$GPU" --cap "$CAP_MAIN" \
        --out "$OUT_BASE/main_adaptive_1_0"

    echo "[main] backbone + adaptive_1.0 done."
}

run_adaptive_sweep() {
    echo ""
    echo "======================================================================"
    echo " ADAPTIVE LAMBDA_SCALE SWEEP  (CAP=$CAP_SWEEP)"
    echo "======================================================================"

    run_adaptive 0.0 "$CAP_SWEEP"
    run_adaptive 0.5 "$CAP_SWEEP"
    run_adaptive 1.0 "$CAP_SWEEP"
    run_adaptive 1.5 "$CAP_SWEEP"
    run_adaptive 2.0 "$CAP_SWEEP"

    echo "[sweep] Adaptive lambda_scale sweep complete."
}

print_summary() {
    echo ""
    echo "======================================================================"
    echo " SUMMARY: SDRi, SI-SDRi, ESTOIi"
    echo "======================================================================"

    echo ""
    echo "--- Main comparison (cap=$CAP_MAIN) ---"
    for tag in main_backbone main_adaptive_1_5; do
        local log="$LOG_DIR/${tag}.log"
        if [[ -f "$log" ]]; then
            overall=$(grep "^\[OVERALL\]" "$log" | tail -1 || echo "(no OVERALL line yet)")
            echo "  [$tag] $overall"
        else
            echo "  [$tag] (not run)"
        fi
    done

    echo ""
    echo "--- Adaptive lambda_scale sweep ---"
    for scale in 0.0 0.5 1.0 1.5 2.0; do
        local tag; tag=$(echo "$scale" | tr '.' '_')
        local log="$LOG_DIR/adaptive_scale_${tag}.log"
        if [[ -f "$log" ]]; then
            overall=$(grep "^\[OVERALL\]" "$log" | tail -1 || echo "(no OVERALL line yet)")
            printf "  scale=%-5s %s\n" "$scale" "$overall"
        else
            printf "  scale=%-5s (not run)\n" "$scale"
        fi
    done
}

# Entry point 
MODE="${1:-all}"

case "$MODE" in
    main)
        run_main
        print_summary
        ;;
    adaptive)
        run_adaptive_sweep
        print_summary
        ;;
    summary)
        print_summary
        ;;
    all)
        run_main
        run_adaptive_sweep
        print_summary
        ;;
    *)
        echo "Usage: bash $0 [main|adaptive|summary|all]"
        exit 1
        ;;
esac
