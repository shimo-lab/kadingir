
library(Rcpp)
library(RcppEigen)

source("src/kadingir.R", chdir = TRUE)
sourceCpp("src/cllsa_core.cpp", rebuild = TRUE)
load("res_cleigenwords_20160119.Rdata")


res.cllsa <- CLLSA(r$corpus.concated, r$document.id.concated,
                   r$sizes.vocabulary, r$lengths.sentence,
                   dim_common_space = 100)

representations.word <- res.cllsa$word_representations
rownames(representations.word) <- c(paste0("(es)", r$vocab.words[[1]]),
                                    paste0("(en)", r$vocab.words[[2]]))

MostSimilar(representations.word, row.names(representations.word),
            positive = c("(en)he"), distance = "cosine", language.search = "en")

save(res.cllsa, file = "res_cllsa.Rdata")
