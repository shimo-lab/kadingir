
source("src/kadingir.R", chdir = TRUE)
source("src/cleigenwords.R", chdir = TRUE)

paths.corpus <- c("data/europarl/europarl-v7.es-en.es.tokenized",
                  "data/europarl/europarl-v7.es-en.en.tokenized")

r <- CLEigenwords(paths.corpus, sizes.vocabulary = c(20000, 20000),
                  dim.common = 100,
                  sizes.window = c(2, 2), aliases.languages=c("es", "en"),
                  plot = TRUE,
                  weighting_tf = FALSE,
                  weight.vsdoc = c(1.0, 1.0))

save(r, file = "res_cleigenwords.Rdata")

p <- r$svd$p_head_domains
V1 <- r$svd$V[p[1]:p[2], ]
V2 <- r$svd$V[(p[3]+1):p[4], ]
vocab.words1 <- paste0("(es)", r$vocab.words[[1]])
vocab.words2 <- paste0("(en)", r$vocab.words[[2]])

V <- rbind(V1, V2)
vocab.words <- c(vocab.words1, vocab.words2)

MostSimilar(V, vocab.words, positive=c("(en)he"), distance = "cosine", language.search = "en")
MostSimilar(V, vocab.words, positive=c("(en)he"), distance = "cosine", language.search = "es")
MostSimilar(V, vocab.words, positive=c("(es)yo"), distance = "cosine", language.search = "en")
MostSimilar(V, vocab.words, positive=c("(es)yo"), distance = "cosine", language.search = "es")
MostSimilar(V, vocab.words, positive=c("(en)him", "(es)yo"), negative=c("(en)he"), distance = "cosine")


TestGoogleTasks(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]], n.cores = 12)
TestWordsim353(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]])