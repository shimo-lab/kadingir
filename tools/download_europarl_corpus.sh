#!/bin/sh

# bash download_corpora.sh

cd ./../corpora

# Download Europarl corpora (v7) from http://www.statmt.org/europarl/
mkdir europarl
cd europarl
wget -i ./../../tools/europarl-v7_list_corpus.txt
ls *.tgz | xargs -n1 tar xzvf
cd -
