#!/bin/bash
nohup python -u -m experiments.$1.$2.$3.experiment > experiments/$1/$2/$3/log/console_output.log &
tail -f experiments/$1/$2/$3/log/console_output.log