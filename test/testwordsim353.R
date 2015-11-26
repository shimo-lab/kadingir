source("eigenwords.R")


## Load data
load("res_eigenwords.Rdata")
vocab <- res.eigenwords$vocab.words
vectors <- res.eigenwords$svd$U
rownames(vectors) <- vocab


TestWordsim353(vectors)
