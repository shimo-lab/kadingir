
set.seed(0)

source("eigenwords.R")


## Tuning parameters
n.vocabulary <- 100 # 語彙に含める単語数
dim.internal <- 20   # 共通空間の次元
window.size <- 2      # 前後何個の単語をcontextとするか
mode <- "oscca"
#path.corpus <- "data/reuters/reuters_rcv1_text.csv"
path.corpus <- "data/train.txt"


res.eigendocs <- Eigendocs(path.corpus, n.vocabulary, dim.internal, window.size, mode = mode)

## Check vector representations
id <- res.eigendocs$document_id
MostSimilar(res.eigendocs$svd$document_vector, id,
            positive=c(id[30]), distance = "cosine", topn = 3)
