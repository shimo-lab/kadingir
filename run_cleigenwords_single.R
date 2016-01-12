
source("src/kadingir.R", chdir = TRUE)
source("src/cleigenwords.R", chdir = TRUE)

paths.corpus <- c("data/europarl/europarl-v7.de-en.en.tokenized")

res.cleigenwords <- CLEigenwords(paths.corpus, max.vocabulary=10000, dim.internal=50,
                                 window.sizes = c(2), aliases=c("en"), plot = TRUE)

save(res.cleigenwords, file = "res_cleigenwords.Rdata")
load("res_cleigenwords.Rdata")

pp <- res.cleigenwords$svd$p_head_domains
V.en <- res.cleigenwords$svd$V[(pp[1]+1):pp[2], ]
vocab.sizes <- res.cleigenwords$vocab.sizes
vocab.words.en <- res.cleigenwords$vocab.words[[1]]


MostSimilar(V.en, vocab.words.en,
            positive=c("he"), distance = "cosine")

MostSimilar(V.en, vocab.words.en,
            positive=c("woman"), distance = "cosine")

TestGoogleTasks(V.en, vocab.words.en, n.cores = 10)
TestWordsim353(V.en, vocab.words.en)
