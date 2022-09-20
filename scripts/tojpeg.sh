#!/bin/bash
pnmtojpeg -qtable=data/qtable.txt $1 ${1/pgm/png} 
