#!/bin/bash
cd "$(dirname "$0")"
ulimit -n 500000
if [ -z "$1" ]; then
  PYTHONPATH=./src/ nice python qald-eval-newpipeline.py --roundtimeout 60 --totaltimeout 29376 &> $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-train-new.txt &
  watch --color "rg --color always 'Intermediate|Error' $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-train-new.txt | tail -n 42"
  wait $!
else
  PYTHONPATH=./src/ nice python qald-eval-newpipeline.py --roundtimeout 60 --totaltimeout 29376 --exp $1 &> $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-train-new-$1.txt &
  watch --color "rg --color always 'Intermediate|Error' $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-train-new-$1.txt | tail -n 42"
  wait $!
fi

#watch "tail -n 30 $(date '+%Y-%m-%d')-qald-eval-train-parallel.txt"
