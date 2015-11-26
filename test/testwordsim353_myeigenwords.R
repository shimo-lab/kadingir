source("eigenwords.R")


## Load data
load("res_eigenwords.Rdata")
vocab <- res.eigenwords$vocab.words
vectors <- res.eigenwords$svd$U
rownames(vectors) <- vocab



## Calculate similarities and correlation
wordsims <- read.csv("data/wordsim353/combined.csv", header = 1)

cosine.similarity <- function (w1, w2) {
  v1 <- vectors[w1, ]
  v2 <- vectors[w2, ]
  
  (v1 %*% v2) / sqrt((v1 %*% v1) * (v2 %*% v2))
}

n.tests <- nrow(wordsims)
similarity.eigenwords <- rep(NULL, times = n.tests)

for(i in seq(n.tests)) {
  w1 <- as.character(wordsims[i, 1])
  w2 <- as.character(wordsims[i, 2])
  
  if (w1 %in% vocab && w2 %in% vocab) {
    similarity.eigenwords[i] <- cosine.similarity(w1, w2)
  }
  print(w)
  print(similarity.eigenwords[i])
  cat("\n\n")
}
similarity.human <- wordsims[ , 3]

print(!is.na(similarity.eigenwords))

similarity.human <- similarity.human[!is.na(similarity.eigenwords)]
similarity.eigenwords <- similarity.eigenwords[!is.na(similarity.eigenwords)]

cor(similarity.human, similarity.eigenwords, method = "spearman")
plot(similarity.human, similarity.eigenwords)
