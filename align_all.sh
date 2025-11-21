#!/usr/bin/env bash

set -e

MODELS_DIR="models"
OUT_DIR="aligned_models"
DICT="alignment_dict.txt"

mkdir -p "$OUT_DIR"

for sub in "$MODELS_DIR"/*; do
    if [ -d "$sub" ]; then
        subname=$(basename "$sub")

        SRC="$sub/group0.vec"
        TRG="$sub/group1.vec"

        # 跳过不存在 group0 vec 或 group1 vec 的文件夹
        if [ ! -f "$SRC" ] || [ ! -f "$TRG" ]; then
            echo "Skipping $subname (missing group0.vec or group1.vec)"
            continue
        fi

        echo "Aligning: $subname"

        # 输出子目录
        mkdir -p "$OUT_DIR/$subname"

        OUT_SRC="$OUT_DIR/$subname/group0_mapped.vec"
        OUT_TRG="$OUT_DIR/$subname/group1_mapped.vec"

        python3 map_embeddings.py \
            --supervised "$DICT" \
            "$SRC" \
            "$TRG" \
            "$OUT_SRC" \
            "$OUT_TRG"

        echo "Done: $subname → $OUT_DIR/$subname"
    fi
done

echo "All models aligned successfully!"
