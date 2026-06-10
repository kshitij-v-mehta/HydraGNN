#!/bin/bash
set -x

time wget https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/neutral_train.tar.gz
time tar -xf neutral_train.tar.gz
