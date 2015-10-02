
source("eigenwords.R")

## Tuning parameters
min.count <- 100      # 出現回数がmin.count回以下の単語はvocabに入れない
dim.internal <- 200   # 共通空間の次元
window.size <- 2      # 前後何個の単語をcontextとするか

## Making train data
f <- file("data/text8", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig <- unlist(strsplit(tolower(line), " "))
rm(line)

## Eigenwords
res.eigenwords <- Eigenwords(sentence.orig, min.count, dim.internal, window.size)

## Check vector representations
MostSimilar(res.eigenwords, positive=c("man"), topn=10)
MostSimilar(res.eigenwords, positive=c("king", "woman"), negative=c("man"),
            normalize=TRUE, format="cosine", topn=10)
