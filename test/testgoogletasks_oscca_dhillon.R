library(data.table)

source("eigenwords.R")



n.use.vocab <- 30000

vectors <- fread("data/rcv1.tscca.100k.200.c10.vectors", sep = " ", header = FALSE)
vectors <- as.matrix(vectors)
vocab <- readLines("data/rcv1.tscca.100k.200.c10.vocab")

vectors <- vectors[seq(2, n.use.vocab), ]
vocab <- vocab[seq(2, n.use.vocab)]

MostSimilar(vectors, vocab, positive = c("of"), distance = "cosine")
MostSimilar(vectors, vocab, positive = c("Japan"), distance = "cosine")

TestGoogleTasks(vectors, vocab, "test/questions-words.txt", n.cores = 24)
