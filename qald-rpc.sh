#!/bin/bash
cd "$(dirname "$0")"
ulimit -n 500000
PYTHONPATH=./src/ nice python qald-rpc-service.py

