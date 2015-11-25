source("eigenwords.R")


n.use.vocab <- 100001

load("res_eigenwords.Rdata")
vocab <- res.eigenwords$vocab.words[1:n.use.vocab]
vectors <- res.eigenwords$svd$U[1:n.use.vocab, ]
rownames(vectors) <- vocab

wordsims <- read.csv("data/wordsim353/combined.csv", header = 1)

cosine.similarity <- function (w1, w2) {
  v1 <- vectors[w1, ]
  v2 <- vectors[w2, ]
  
  (v1 %*% v2) / sqrt((v1 %*% v1) * (v2 %*% v2))
}

n.tests <- nrow(wordsims)
similarity.eigenwords <- rep(0, times = n.tests)

for(i in seq(n.tests)) {
  w <- wordsims[i, ]
  similarity.eigenwords[i] <- cosine.similarity(w[[1]], w[[2]])
}
similarity.human <- wordsims[ , 3]

cor(similarity.human, similarity.eigenwords, method = "spearman")
plot(similarity.human, similarity.eigenwords)
