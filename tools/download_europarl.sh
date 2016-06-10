#!/bin/sh

# bash download_corpora.sh

cd ./../corpora

# Download Europarl corpora (v7) from http://www.statmt.org/europarl/
mkdir europarl
cd europarl
wget -i ./../../tools/url_list_europarl-v7.txt -w 10 --no-clobber
ls *.tgz | xargs -n1 tar xzvf
cd -
