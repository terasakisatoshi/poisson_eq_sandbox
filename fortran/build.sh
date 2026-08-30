#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
gfortran -O3 -o poisson main.f90
