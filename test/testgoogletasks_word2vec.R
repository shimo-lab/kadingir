source("eigenwords.R")


n.use.vocab <- 80000

table <- read.table("./../word2vec/vectors.txt", sep = " ", skip = 1)
table <- table[1:n.use.vocab, ]
vocab <- as.vector(table[ , 1])
vectors <- as.matrix(table[ , 2:201])

MostSimilar(vectors, vocab, positive = c("japan"), distance = "cosine")

TestGoogleTasks(vectors, vocab, "questions-words.txt", n.cores = 12)
