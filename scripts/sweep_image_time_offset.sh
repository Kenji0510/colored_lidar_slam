#!/usr/bin/env bash
# 画像タイムスタンプのオフセットを掃引し、各実行の評価値を 1 つの表にまとめる。
#
# 使い方:
#   scripts/sweep_image_time_offset.sh
#   LIDAR=avia OFFSETS_MS="80 90 100 110 120" scripts/sweep_image_time_offset.sh
#   EXTRA_ARGS="--no-visual-alignment" scripts/sweep_image_time_offset.sh
#
# 各実行の結果は `<SAVE_ROOT_DIR>/<lidar>_imgoff_<offset>ms/` に保存されるので、
# 既定の出力先 `<SAVE_ROOT_DIR>/<lidar>/` は上書きしない。
# ログと集計表は data/output/offset_sweep/<日時>/ に保存する。
#
# 見方（すべて小さいほど良い。時刻が合うほど、補正前から輝度が合い、画像の補正も小さくなる）:
#   lidar_rmse_mean_m         画像位置合わせ後の姿勢で次フレーム以降の ICP が合っているか
#   image_initial_rmse_mean   画像位置合わせ「前」の輝度 RMSE
#   correction_t/r_median     実際に反映された画像の補正量
set -euo pipefail

cd "$(dirname "$0")/.."

LIDAR="${LIDAR:-avia}"
OFFSETS_MS="${OFFSETS_MS:-0 50 60 70 80 90 100 110 120 130 140 150}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_DIR="data/output/offset_sweep/$(date +%Y%m%d-%H%M%S)"
SUMMARY="$RUN_DIR/summary.tsv"

mkdir -p "$RUN_DIR"
cargo build --release --bin re_lidar_slam

header_written=0
for ms in $OFFSETS_MS; do
    sec=$(awk -v ms="$ms" 'BEGIN { printf "%.6f", ms / 1000.0 }')
    tag="imgoff_${ms/#-/m}ms"
    log="$RUN_DIR/${tag}.log"
    echo "=== image time offset ${ms} ms (${sec} s) -> $log"

    # shellcheck disable=SC2086  # EXTRA_ARGS は意図的に単語分割する
    if ! RUST_LOG=info ./target/release/re_lidar_slam \
        --lidar "$LIDAR" --image-time-offset "$sec" --output-tag "$tag" $EXTRA_ARGS \
        >"$log" 2>&1; then
        echo "    run failed (see $log)"
        continue
    fi

    line=$(grep -h "Image time offset evaluation:" "$log" | tail -n 1 | sed 's/.*evaluation: //')
    if [[ -z "$line" ]]; then
        echo "    no evaluation line found (see $log)"
        continue
    fi
    if [[ $header_written -eq 0 ]]; then
        echo "$line" | tr ' ' '\n' | cut -d= -f1 | paste -sd '\t' >"$SUMMARY"
        header_written=1
    fi
    echo "$line" | tr ' ' '\n' | cut -d= -f2 | paste -sd '\t' >>"$SUMMARY"
done

if [[ -f "$SUMMARY" ]]; then
    echo
    echo "Summary: $SUMMARY"
    column -t -s $'\t' "$SUMMARY"
fi
