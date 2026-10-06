#!/usr/bin/env sh
# Portable launcher for Fayd Al-Ruh.
#
# Prefers a frozen executable sitting next to this script (a built bundle);
# otherwise falls back to running the source checkout, so the same script works
# during development.
set -eu

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if [ -x "$here/fayd-al-ruh" ]; then
    exec "$here/fayd-al-ruh" "$@"
fi

exec python3 -m cli "$@"
