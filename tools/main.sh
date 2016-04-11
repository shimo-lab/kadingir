#!/bin/sh

bash download_europarl_corpus.sh
bash download_text.sh

cd word2vec/
make

bash preprocess_europarl.sh
