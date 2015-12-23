
source("kadingir.R")
source("mceigendocs.R")

paths.corpus <- c("data/europarl-v7.de-en.100000.en")

res.eigendocs <- MCEigendocs(paths.corpus, max.vocabulary=10000, dim.internal=50,
                             window.sizes = c(2), aliases=c("en"), plot = TRUE)

p <- res.eigendocs$svd$p_cumsum
V.en <- res.eigendocs$svd$V[p[1]:p[2], ]
vocab.words.en <- res.eigendocs$vocab.words[[1]]

save(res.eigendocs, file = "res_eigendocs.Rdata")


load("res_eigendocs.Rdata")

p <- res.eigendocs$svd$p_cumsum
V.en <- res.eigendocs$svd$V[p[1]:p[2], ]
vocab.words.en <- res.eigendocs$vocab.words[[1]]


MostSimilar(V.en, vocab.words.en,
            positive=c("he"), distance = "cosine")

MostSimilar(V.en, vocab.words.en,
            positive=c("woman"), distance = "cosine")

TestGoogleTasks(V.en, vocab.words.en, n.cores = 10)
TestWordsim353(V.en, vocab.words.en)
