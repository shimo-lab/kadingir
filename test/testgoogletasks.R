source("eigenwords.R")


# my eigenwords
load("res_eigenwords.Rdata")
vocab <- res.eigenwords$vocab.words
vectors <- res.eigenwords$svd$word_vector

# pretrained embedding by Dhillon
vocab <- readLines("data/rcv1.oscca.100k.200.c10.vocab")
vectors <- fread("data/rcv1.oscca.100k.200.c10.vectors", sep = " ", header = FALSE)
vectors <- as.matrix(vectors)

# word2vec
table <- read.table("./../word2vec/vectors.txt", sep = " ", skip = 1)
vocab <- as.vector(table[ , 1])
vectors <- as.matrix(table[ , 2:201])

rownames(vectors) <- vocab
vocab.word2vec <- vocab


for (n.use.vocab in c(10000,20000,40000,60000,80000,100000)) {
  vectors.topn <- vectors[seq(2, n.use.vocab), ]
  vocab.topn <- vocab[seq(2, n.use.vocab)]
  rownames(vectors.topn) <- vocab.topn
  
  print(MostSimilar(vectors.topn, vocab.topn, positive = c("man"), distance = "cosine"))
  print(MostSimilar(vectors.topn, vocab.topn, positive = c("japan"), distance = "cosine"))
  print(MostSimilar(vectors.topn, vocab.topn, positive = c("Japan"), distance = "cosine"))
  
  TestGoogleTasks(vectors.topn, vocab.topn, "test/questions-words.txt", n.cores = 12)
}




# 実験
index.word2vec <- tolower(vocab) %in% vocab.word2vec
vocab <- vocab[index.word2vec]
vectors <- vectors[index.word2vec, ]
rownames(vectors) <- vocab

print(MostSimilar(vectors, vocab, positive = c("man"), distance = "cosine"))

TestGoogleTasks(vectors, vocab, "test/questions-words.txt", n.cores = 32)
