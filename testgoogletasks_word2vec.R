table <- read.table("./../word2vec/vectors.txt", sep = " ", skip = 1)
table <- table[1:30000, ]
vocab <- as.vector(table[ , 1])
vectors <- as.matrix(table[ , 2:201])

MostSimilar(vectors, vocab, positive = c("japan"), distance = "cosine")

TestGoogleTasks(vectors, vocab, "test/questions-words.txt", n.cores = 12, distance = "cosine")

queries <- read.csv("test/questions-words.txt", header = FALSE, sep = " ", comment.char = ":")
distance <- "euclid"
for(i in 10000:10100){
  q <- as.character(unlist(queries[i, ]))
  res.MostSimilar <- MostSimilar(vectors, vocab,
                                 positive=c(q[[2]], q[[3]]), negative=c(q[[1]]),
                                 distance=distance, topn=1, print.error = FALSE)
  if (res.MostSimilar) {
    cat(names(res.MostSimilar), " ", q[[4]], "\n")
  }
}

