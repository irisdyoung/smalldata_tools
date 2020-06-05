#!/bin/bash

source /reg/g/psdm/etc/psconda.sh
ABS_PATH=/reg/g/psdm/sw/tools/smalldata_tools/examples

sbatch --nodes=2 --time=5 $ABS_PATH/smalldata_producer_arp.py "$@"
