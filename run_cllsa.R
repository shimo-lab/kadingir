
library(Rcpp)
library(RcppEigen)

source("src/kadingir.R", chdir = TRUE)
sourceCpp("src/cllsa_core.cpp", rebuild = TRUE)
load("res_cleigenwords_20160120.Rdata")


res.cllsa <- CLLSA(r$corpus.concated, r$document.id.concated,
                   r$sizes.vocabulary, r$lengths.sentence,
                   dim_common_space = 100)

res.cllsa$representations.word <- res.cllsa$word_representations
rownames(res.cllsa$representations.word) <- c(paste0("(es)", r$vocab.words[[1]]),
                                              paste0("(en)", r$vocab.words[[2]]))

MostSimilar(res.cllsa$representations.word, row.names(res.cllsa$representations.word),
            positive = c("(en)he"), distance = "cosine", language.search = "en")

save(res.cllsa, file = "res_cllsa.Rdata")
