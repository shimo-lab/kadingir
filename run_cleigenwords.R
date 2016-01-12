
source("src/kadingir.R", chdir = TRUE)
source("src/cleigenwords.R", chdir = TRUE)

paths.corpus <- c("data/europarl/europarl-v7.de-en.de.tokenized.100000",
                  "data/europarl/europarl-v7.de-en.en.tokenized.100000")

r <- CLEigenwords(paths.corpus, sizes.vocabulary = c(10000, 8000),
                  dim.common = 100,
                  sizes.window = c(2, 2), aliases.languages=c("de", "en"),
                  plot = TRUE,
                  weighting_tf = FALSE,
                  weight.vsdoc = c(1.0, 1.0))

save(r, file = "res_cleigenwords.Rdata")

p <- r$svd$p_head_domains
V.de <- r$svd$V[p[1]:p[2], ]
V.en <- r$svd$V[(p[3]+1):p[4], ]
vocab.words.de <- paste0("(de)", r$vocab.words[[1]])
vocab.words.en <- paste0("(en)", r$vocab.words[[2]])

V <- rbind(V.de, V.en)
vocab.words <- c(vocab.words.de, vocab.words.en)

MostSimilar(V, vocab.words, positive=c("(en)he"), distance = "cosine", language.search = "en")
MostSimilar(V, vocab.words, positive=c("(en)he"), distance = "cosine", language.search = "de")
MostSimilar(V, vocab.words, positive=c("(de)Ich"), distance = "cosine", language.search = "en")
MostSimilar(V, vocab.words, positive=c("(de)Ich"), distance = "cosine", language.search = "de")
MostSimilar(V, vocab.words, positive=c("(en)him", "(de)ich"), negative=c("(en)he"), distance = "cosine")


TestGoogleTasks(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]], n.cores = 24)
TestWordsim353(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]])