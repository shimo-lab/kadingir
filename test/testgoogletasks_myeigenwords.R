source("eigenwords.R")


n.use.vocab <- 100001

load("res_eigenwords.Rdata")
vocab <- res.eigenwords$vocab.words[1:n.use.vocab]
vectors <- res.eigenwords$svd$U[1:n.use.vocab, ]

MostSimilar(vectors, vocab, positive = c("of"), distance = "cosine")
MostSimilar(vectors, vocab, positive = c("Japan"), distance = "cosine")

TestGoogleTasks(vectors, vocab, "test/questions-words.txt", n.cores = 24)
