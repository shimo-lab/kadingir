library(tsne)

load("res_eigenwords.Rdata")

vocab   <- res.eigenwords$vocab.words
vectors <- res.eigenwords$svd$word_vector

begin <- 1
end <- 4000

vectors.tsne <- tsne(vectors[seq(begin, end), ], max_iter=200)
vectors.tsne2 <- vectors.tsne + 0.5*rnorm(2*(end - begin))

pdf(file = "tsne_words.pdf", width = 60, height = 40)
plot(vectors.tsne2, t='n', main="tsne")
text(vectors.tsne2, labels=vocab[seq(begin, end)])
dev.off()
