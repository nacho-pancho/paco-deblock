#!/bin/bash
echo "image: ${1}"
echo "qual.: ${2}"
echo "tau  : ${3}"
echo "lam. : ${4}"

python/paco_deblocking.py data/${1}_q${2}.pgm data/q${2}.txt --tau ${3} --lam ${4} data/${1}.pgm
