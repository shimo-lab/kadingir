#!/bin/sh
 
# preprocess_europarl.sh
# * bash preprocess_europarl.sh de
# * bash preprocess_europarl.sh es
 
workdir='./../corpora/europarl/'
 
# tokenize corpus
corpus=$workdir"europarl-v7."$1"-en."
echo $corpus$1
$workdir"tools/tokenizer.perl" -l $1 < $corpus$1   | awk '{print tolower($0)}' > $corpus$1".tokenized"
echo $corpus"en"
$workdir"tools/tokenizer.perl" -l en < $corpus"en" | awk '{print tolower($0)}' > $corpus"en.tokenized"
 
for path in $corpus??".tokenized"
do
    echo $path
 
    # construct phrase
    ./word2vec/word2phrase -train $path -output $path".phrase1" -threshold 100 -debug 2
    ./word2vec/word2phrase -train $path".phrase1" -output $path".phrase2" -threshold 50 -debug 2
 
    # generate small corpus
    cat $path | awk 'NR <= 10000'  > $path".10000"
    cat $path | awk 'NR <= 100000' > $path".100000"
    cat $path | awk 'NR <= 500000' > $path".500000"
    cat $path".phrase2" | awk 'NR <= 10000'  > $path".phrase2.10000"
    cat $path".phrase2" | awk 'NR <= 100000' > $path".phrase2.100000"
done
