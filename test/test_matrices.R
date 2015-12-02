
source("eigenwords.R")

sentence <- sample(0:6, size = 100, replace = TRUE)
window.size <- 2
n.train.words <- length(sentence)
n.vocab <- length(unique(sentence))
dim.internal <- 20

m <- make.matrices(sentence, window.size, n.train.words, n.vocab, skip.null.words = TRUE)
res.eigenwords <- EigenwordsRedSVD(as.integer(sentence), window.size, n.vocab, dim.internal, mode_oscca = TRUE)


## Test matrices

res.eigenwords$tWW_h
diag(t(m$W) %*% m$W)**(-1/2)

res.eigenwords$tCC_h
diag(t(m$C) %*% m$C)**(-1/2)

res.eigenwords$tWC
t(m$W) %*% m$C

t(m$W) %*% m$W
t(m$C) %*% m$C

all(res.eigenwords$tWC == t(m$W) %*% m$C)
sum((res.eigenwords$tWW_h - diag(t(m$W) %*% m$W)**(-1/2))**2)
sum((res.eigenwords$tCC_h - diag(t(m$C) %*% m$C)**(-1/2))**2)


