#!/bin/sh

cd ./../corpora

# Download Europarl corpora (v7) from http://www.statmt.org/europarl/
mkdir text8
cd text8

wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f

cd -
