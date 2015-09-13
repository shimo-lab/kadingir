install.packages("devtools")

library(devtools)
dev_mode(on=T)
install_github("xiangze/RRedsvd")


library(hash)
library(Matrix)
library(RRedsvd)


min.count <- 2
dim.internal <- 100
window.size <- 2


## make dictionary
f <- file("train.txt", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig.full <- unlist(strsplit(tolower(line), " "))
sentence.orig <- sentence.orig.full[1:100000]
vocab.orig <- unique(sentence.orig)

sentence <- match(sentence.orig, vocab.orig)
n.train.words <- length(sentence)

rm(sentence.orig, sentence.orig.full)


if (min.count > 0){
    d.table <- table(sentence)
    vocab <- names(d.table[d.table >= min.count])
} else {
    vocab <- unique(sentence)
}

vocab <- as.numeric(vocab)
n.vocab <- length(vocab)

## make hash table
##   word hash
##     -> index that indicates row word vector of representation matrix
word2index <- hash()
for(i in seq(n.vocab)){
    word2index[as.character(vocab[i])] <- i
}

W <- Matrix(0, nrow = n.train.words, ncol = n.vocab, sparse = TRUE)
#C <- Matrix(0, nrow = n.train.words, ncol = 2*window.size*n.vocab, sparse = TRUE)

for(i.sentence in seq(sentence)){
    word <- sentence[i.sentence]
    index <- as.character(word)

    if(has.key(index, word2index)){
        i.vocab <- word2index[[index]]
        W[i.sentence, i.vocab] <- 1
    }
}

c.list <- c()
C <- 0
for(i.context in sort(c(seq(window.size), -seq(window.size)), decreasing = TRUE)){
    c.temp <- Matrix(0, nrow = n.train.words, ncol = n.vocab, sparse = TRUE)

    c.row.start <- max(i.context + 1, 1)
    c.row.end <- min(n.train.words + i.context, n.train.words)
    w.row.start <- max(1 - i.context, 1)
    w.row.end <- min(n.train.words - i.context, n.train.words)

    print(c(c.row.start, c.row.end, w.row.start, w.row.end))
    
    c.temp[c.row.start:c.row.end, ] <- W[w.row.start:w.row.end, ]
    c.list <- c(c.list, c.temp)

    if(is.null(dim(C))){
        C <- c.temp
    }else{
        C <- cbind(C, c.temp)
    }
}

Cww <- t(W) %*% W
Cwc <- t(W) %*% C
Ccc <- t(C) %*% C

A <- Diagonal(ncol(W), diag(Cww)^(-1/2)) %*% Cwc %*% Diagonal(ncol(C), diag(Ccc)^(-1/2))
redsvd.A <- redsvd(A, dim.internal)
