#!/bin/bash

trap 'echo Error: $0:$LINENO stopped; exit 1' ERR INT
set -eu

echo "[`date "+%Y/%m/%d %H:%M:%S"`]make init start"
bash "$MAKE_PATH"/tools/init/dir_setup.sh
echo "[`date "+%Y/%m/%d %H:%M:%S"`]make init end"
