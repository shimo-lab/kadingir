#!/bin/sh


for path in './../corpora/europarl/europarl-v7.merged.'??;
do
    lang="${path##*.}"
    ./../corpora/europarl/tools/tokenizer.perl -l $lang < $path | awk '{print tolower($0)}' > $path".tokenized"
    cat $path".tokenized" | awk 'NR <= 10000' > $path".tokenized.10000"
    cat $path".tokenized" | awk 'NR <= 100000' > $path".tokenized.100000"
done
