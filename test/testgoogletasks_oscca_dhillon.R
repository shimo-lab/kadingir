source("eigenwords.R")


n.use.vocab <- 10000

table <- read.table("data/rcv1.tscca.100k.200.c10", sep = " ")
table <- table[1:n.use.vocab, ]
vocab <- as.vector(table[ , 1])
vectors <- as.matrix(table[ , 2:201])
class(vectors) <- "numeric"

MostSimilar(vectors, vocab, positive = c("of"), distance = "cosine")
MostSimilar(vectors, vocab, positive = c("Japan"), distance = "cosine")

TestGoogleTasks(vectors, vocab, "test/questions-words.txt", n.cores = 12)
