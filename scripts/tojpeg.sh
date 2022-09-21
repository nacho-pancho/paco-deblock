#!/bin/bash
# to see the qtables:
pnmtojpeg -quality=30 $1 > ${1/.pgm/_q30.jpg} 
jpegtopnm -tracelevel 10 ${1/.pgm/_q30.jpg} > ${1/.pgm/_q30.pgm} 2> /tmp/pepe.txt
grep -A8 "Define Quanti" /tmp/pepe.txt | tail -n 8 > /tmp/q.txt
# to use a custom qtable
# pnmtojpeg -qtable=data/qtable.txt $1 ${1/pgm/png} 
