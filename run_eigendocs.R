
set.seed(0)

source("eigenwords.R")


## Tuning parameters
n.vocabulary <- 100000 # 語彙に含める単語数
dim.internal <- 200   # 共通空間の次元
window.size <- 4      # 前後何個の単語をcontextとするか
mode <- "oscca"
path.corpus <- "data/reuters/reuters_rcv1_text100000.csv"
#path.corpus <- "data/train.txt"


res.eigendocs <- Eigendocs(path.corpus, n.vocabulary, dim.internal, window.size, mode = mode)
save(res.eigendocs, file = "res_eigendocs.Rdata")

## Check vector representations
id <- res.eigendocs$document_id
MostSimilar(res.eigendocs$svd$document_vector, id,
            positive=c(id[103]), distance = "cosine", topn = 3)


## Check vector representations
MostSimilar(res.eigendocs$svd$word_vector, res.eigendocs$vocab.words,
            positive=c("man"), distance = "cosine")
MostSimilar(res.eigendocs$svd$word_vector, res.eigendocs$vocab.words,
            positive=c("king", "woman"), negative=c("man"), distance = "cosine")

## Test some tasks for check
TestGoogleTasks(res.eigendocs$svd$word_vector, res.eigendocs$vocab.words, n.cores = 24)
TestWordsim353(res.eigendocs$svd$word_vector, res.eigendocs$vocab.words)
