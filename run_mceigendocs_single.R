
source("kadingir.R")
source("mceigendocs.R")

paths.corpus <- c("data/europarl/europarl-v7.de-en.en")

res.mceigendocs <- MCEigendocs(paths.corpus, max.vocabulary=10000, dim.internal=50,
                               window.sizes = c(2), aliases=c("en"), plot = TRUE)

save(res.mceigendocs, file = "res_mceigendocs.Rdata")
load("res_mceigendocs.Rdata")

pp <- res.mceigendocs$svd$p_head_domains
V.en <- res.mceigendocs$svd$V[(pp[1]+1):pp[2], ]
vocab.sizes <- res.mceigendocs$vocab.sizes
vocab.words.en <- res.mceigendocs$vocab.words[[1]]


MostSimilar(V.en, vocab.words.en,
            positive=c("he"), distance = "cosine")

MostSimilar(V.en, vocab.words.en,
            positive=c("woman"), distance = "cosine")

TestGoogleTasks(V.en, vocab.words.en, n.cores = 10)
TestWordsim353(V.en, vocab.words.en)
