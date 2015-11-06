
set.seed(0)

source("eigenwords.R")


## Tuning parameters
min.count <- 200      # 出現回数がmin.count回以下の単語はvocabに入れない
dim.internal <- 200   # 共通空間の次元
window.size <- 4      # 前後何個の単語をcontextとするか


## Making train data
f <- file("data/enwiki1GB.txt", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig <- unlist(strsplit(tolower(line), " "))
rm(line)


## Eigenwords
res.eigenwords <- Eigenwords(sentence.orig, min.count, dim.internal, window.size, use.block.matrix=FALSE)
save(res.eigenwords, file = "res_eigenwords.Rdata")


## Check vector representations
MostSimilar(res.eigenwords, positive=c("man"))
MostSimilar(res.eigenwords, positive=c("king", "woman"), negative=c("man"))

## Calcurate accuracy of Google analogy task
TestGoogleTasks(res.eigenwords, "test/questions-words.txt")
