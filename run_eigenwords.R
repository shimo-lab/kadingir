
source("eigenwords.R")

## Tuning parameters
min.count <- 10       # 出現回数がmin.count回以下の単語はvocabに入れない
dim.internal <- 200   # 共通空間の次元
window.size <- 2      # 前後何個の単語をcontextとするか

## Making train data
f <- file("data/text8", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig.full <- unlist(strsplit(tolower(line), " "))
sentence.orig <- sentence.orig.full[1:1000000]
vocab.orig <- unique(sentence.orig)

## Eigenwords
res.eigenwords <- eigenwords(sentence.orig, vocab.orig, min.count, dim.internal, window.size)

## Check vector representations
most.similar(res.eigenwords, positive=c("統計"), topn=10)
