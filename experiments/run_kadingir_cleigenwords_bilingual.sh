#!/bin/sh

cd ./../cpp

time make -B kadingir_cleigenwords_bilingual

time ./kadingir_cleigenwords_bilingual \
    --corpus1 ./../corpora/europarl/europarl-v7.es-en.es.tokenized.500000 \
    --corpus2 ./../corpora/europarl/europarl-v7.es-en.en.tokenized.500000 \
    --output output_kadingir.txt \
    --vocab1 10000 \
    --vocab2 10000 \
    --window1 2 \
    --window2 2 \
    --dim 40

cd -
