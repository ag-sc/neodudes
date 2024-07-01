#!/bin/bash
cd "$(dirname "$0")"
ulimit -n 500000
#10800 = 3h, 7200 = 2h, 3600 = 1h
if [ -z "$1" ]; then
  PYTHONPATH=./src/ nice python qald-eval-newpipeline.py --test --roundtimeout 60 --totaltimeout 10800 &> $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-test-new.txt &
  watch --color "rg --color always 'Intermediate|Error' $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-test-new.txt | tail -n 20"
  wait $!
else
  PYTHONPATH=./src/ nice python qald-eval-newpipeline.py --test --roundtimeout 60 --totaltimeout 10800 --exp $1 &> $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-test-new-$1.txt &
  watch --color "rg --color always 'Intermediate|Error' $(date '+%Y-%m-%d')-qald-eval-$(cat /etc/hostname)-test-new-$1.txt | tail -n 20"
  wait $!
fi

