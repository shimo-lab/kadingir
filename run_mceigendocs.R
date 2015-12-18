
source("mceigendocs.R")

paths.corpus <- c("~/data/corpus/europarl/europarl-v7.de-en.10000.de",
                  "~/data/corpus/europarl/europarl-v7.de-en.10000.en")

r <- MCEigendocs(paths.corpus, max.vocabulary=1000, dim.internal=50,
                 window.sizes = c(2, 3), aliases=c("de", "en"), plot = FALSE)
