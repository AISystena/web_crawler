#!/bin/bash

trap 'echo Error: $0:$LINENO stopped; exit 1' ERR INT
set -eu

bash "$CURRENT_PATH"/tools/clean/output_clear.sh
