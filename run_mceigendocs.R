
source("mceigendocs.R")

paths.corpus <- c("data/europarl/europarl-v7.de-en.10000.de",
                  "data/europarl/europarl-v7.de-en.10000.en")


MCEigendocs(paths.corpus, max.vocabulary=100, dim.internal=50,
            window.sizes = c(2, 3), aliases=c("de", "en"))
