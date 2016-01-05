
source("src/kadingir.R", chdir = TRUE)
source("src/mceigendocs.R", chdir = TRUE)

paths.corpus <- c("data/europarl/europarl-v7.de-en.de.tokenized.100000",
                  "data/europarl/europarl-v7.de-en.en.tokenized.100000")

r <- MCEigendocs(paths.corpus, max.vocabulary=10000, dim.internal=100,
                 window.sizes = c(2, 3), aliases=c("de", "en"),
                 plot = TRUE,
                 weighting_tf = FALSE ,
                 weight_vsdoc = c(0.1, 0.1))

p <- r$svd$p_head_domains
V.de <- r$svd$V[p[1]:p[2], ]
V.en <- r$svd$V[(p[3]+1):p[4], ]
vocab.words.de <- paste0("(de)", r$vocab.words[[1]])
vocab.words.en <- paste0("(en)", r$vocab.words[[2]])

V <- rbind(V.de, V.en)
vocab.words <- c(vocab.words.de, vocab.words.en)

MostSimilar(V, vocab.words, positive=c("(en)he"), distance = "cosine")
MostSimilar(V, vocab.words, positive=c("(en)who"), distance = "cosine")
MostSimilar(V, vocab.words, positive=c("(de)Ich"), distance = "cosine")

MostSimilar(V, vocab.words,
            positive=c("king", "woman"), negative=c("man"), distance = "cosine")


TestGoogleTasks(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]], n.cores = 24)
TestWordsim353(r$svd$V[(p[3]+1):p[4], ], r$vocab.words[[2]])