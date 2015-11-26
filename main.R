
set.seed(0)

source("eigenwords.R")


## Tuning parameters
n.vocabulary <- 100000 # 語彙に含める単語数
dim.internal <- 200   # 共通空間の次元
window.size <- 4      # 前後何個の単語をcontextとするか
path.corpus <- "data/reuters/reuters_rcv1_text.csv"
#path.corpus <- "data/text8"


res.eigenwords <- Eigenwords(path.corpus, n.vocabulary, dim.internal, window.size,
                             mode = "oscca", use.block.matrix=FALSE)
save(res.eigenwords, file = "res_eigenwords.Rdata")


## Check vector representations
MostSimilar(res.eigenwords$svd$U, res.eigenwords$vocab.words, positive=c("man"), distance = "cosine")
MostSimilar(res.eigenwords$svd$U, res.eigenwords$vocab.words, positive=c("king", "woman"), negative=c("man"), distance = "cosine")

## Calcurate accuracy of Google analogy task
TestGoogleTasks(res.eigenwords$svd$U, res.eigenwords$vocab.words,
                "test/questions-words.txt", n.cores = 12)

