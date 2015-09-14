
## Eigenwords

library(hash)
library(Matrix)
library(RRedsvd)
library(tcltk)


min.count <- 10       # 出現回数が10回以下の単語はvocabに入れない
dim.internal <- 200   # 共通空間の次元
window.size <- 2      # 前後何個の単語をcontextとするか


## make dictionary
f <- file("train_all.txt", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig.full <- unlist(strsplit(tolower(line), " "))
sentence.orig <- sentence.orig.full
vocab.orig <- unique(sentence.orig)

sentence <- match(sentence.orig, vocab.orig)
n.train.words <- length(sentence)

if (min.count > 0){
    d.table <- table(sentence)
    vocab.words <- names(d.table[d.table >= min.count])
} else {
    vocab.words <- unique(sentence)
}

vocab <- as.numeric(vocab.words)
n.vocab <- length(vocab)

## make hash table
##   word hash
##     -> 対応するWの行のインデックス
word2index <- hash()
for(i in seq(n.vocab)){
    word2index[as.character(vocab[i])] <- i
}

W <- Matrix(0, nrow = n.train.words, ncol = n.vocab, sparse = TRUE)

cat("Making matrix W...")
indices <- matrix(0, nrow = length(sentence), ncol = 2)
pb <- txtProgressBar(min = 1, max = length(sentence), style = 3)
for(i.sentence in seq(sentence)){
    
    word <- sentence[i.sentence]
    index <- as.character(word)

    if(has.key(index, word2index)){
        i.vocab <- word2index[[index]]
        indices[i.sentence, ] <- c(i.sentence, i.vocab)
    }

    setTxtProgressBar(pb, i.sentence)
}

indices <- indices[rowSums(indices) > 0, ]
W <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                  x = rep(1, times = nrow(indices)),
                  dims = c(n.train.words, n.vocab))

C <- 0
for(i.context in sort(c(seq(window.size), -seq(window.size)), decreasing = TRUE)){
    c.temp <- Matrix(0, nrow = n.train.words, ncol = n.vocab, sparse = TRUE)

    c.row.start <- max(i.context + 1, 1)
    c.row.end <- min(n.train.words + i.context, n.train.words)
    w.row.start <- max(1 - i.context, 1)
    w.row.end <- min(n.train.words - i.context, n.train.words)
  
    c.temp[c.row.start:c.row.end, ] <- W[w.row.start:w.row.end, ]

    if(is.null(dim(C))){
        C <- c.temp
    }else{
        C <- cbind(C, c.temp)
    }

    print(c(i.context, c.row.start, c.row.end, w.row.start, w.row.end))
}

Cww <- t(W) %*% W
Cwc <- t(W) %*% C
Ccc <- t(C) %*% C

A <- Diagonal(nrow(Cww), diag(Cww)^(-1/2)) %*% Cwc %*% Diagonal(nrow(Ccc), diag(Ccc)^(-1/2))
redsvd.A <- redsvd(A, dim.internal)



######### Check vector representations ##########
most.similar <- function(query, V, rep.vocab, topn = 10){
    if (!query %in% V){
        print(paste0("Error: `", query, "` is not in V."))
        return(FALSE)
    }

    index.query <- which(V == query)
    rep.query <- rep.vocab[index.query, ]
    rep.query.matrix <- matrix(rep.query, nrow=length(vocab), ncol=length(rep.query), byrow=TRUE)
    distances <- sqrt(rowSums((rep.vocab - rep.query.matrix)**2))
    names(distances) <- V
    
    return(sort(distances)[1:topn])
}

most.similar("プログラミング", vocab.orig[vocab], redsvd.A$U)
