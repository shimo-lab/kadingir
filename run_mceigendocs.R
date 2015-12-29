
source("src/kadingir.R", chdir = TRUE)
source("src/mceigendocs.R", chdir = TRUE)

paths.corpus <- c("~/corpus/europarl/europarl-v7.de-en.de",
                  "~/corpus/europarl/europarl-v7.de-en.en")

r <- MCEigendocs(paths.corpus, max.vocabulary=10000, dim.internal=100,
                 window.sizes = c(2, 3), aliases=c("de", "en"), plot = TRUE)

p <- r$svd$p_cumsum
V.de <- r$svd$V[p[1]:p[2], ]
V.en <- r$svd$V[(p[3]+1):p[4], ]
vocab.words.de <- r$vocab.words[[1]]
vocab.words.en <- r$vocab.words[[2]]

V <- rbind(V.de, V.en)
vocab.words <- c(vocab.words.de, vocab.words.en)

MostSimilar(V, vocab.words,
            positive=c("he"), distance = "cosine")

MostSimilar(V, vocab.words,
            positive=c("king", "woman"), negative=c("man"), distance = "cosine")
