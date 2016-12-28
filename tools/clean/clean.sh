#!/bin/bash

trap 'echo Error: $0:$LINENO stopped; exit 1' ERR INT
set -eu

echo "[`date "+%Y/%m/%d %H:%M:%S"`]make clean start"
bash "$MAKE_PATH"/tools/clean/output_clear.sh
echo "[`date "+%Y/%m/%d %H:%M:%S"`]make clean end"
