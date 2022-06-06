#!/bin/bash

source activate thermodrift

set -x

conda list >> mods.txt
