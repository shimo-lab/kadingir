## Implementation of Eigenwords

library(Matrix)
library(RRedsvd)
library(tcltk)


## CCA using randomized SVD
cca.redsvd <- function(W, C, k){
    Cww <- t(W) %*% W
    Cwc <- t(W) %*% C
    Ccc <- t(C) %*% C
    
    A <- Diagonal(nrow(Cww), diag(Cww)^(-1/2)) %*% Cwc %*% Diagonal(nrow(Ccc), diag(Ccc)^(-1/2))
    
    return(redsvd(A, k))
}


eigenwords <- function(sentence.orig, vocab.orig, min.count = 10,
                       dim.internal = 200, window.size = 2){
    if (min.count > 0){
        d.table <- table(sentence.orig)
        vocab.words <- names(d.table[d.table >= min.count])
    } else {
        vocab.words <- unique(sentence)
    }

    sentence <- match(sentence.orig, vocab.words, nomatch = 0)
    n.vocab <- length(vocab.words)
    n.train.words <- length(sentence)


    ## Calculate Eigenwords
    ##  行列W, Cを構成する
    ##  実行速度の観点から，1が立つ要素のインデックスをfor文で生成し，
    ##  sparseMatrix関数を使ってまとめて行列を生成している．
    
    ## W を構成
    cat("Constructing W")
    pb <- txtProgressBar(min = 1, max = length(sentence), style = 3)
    indices <- matrix(0, nrow = length(sentence), ncol = 2)
    for(i.sentence in seq(sentence)){    
        word <- sentence[i.sentence]

        if(word != 0){
            indices[i.sentence, ] <- c(i.sentence, word)
        }

        setTxtProgressBar(pb, i.sentence)
    }
    
    indices <- indices[rowSums(indices) > 0, ]
    W <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                      x = rep(1, times = nrow(indices)),
                      dims = c(n.train.words, n.vocab))
    
    ## C を構成
    cat("Constructing C")
    pb <- txtProgressBar(min = 1, max = length(sentence), style = 3)
    indices <- matrix(0, nrow = 2*window.size*length(sentence), ncol = 2)
    offsets <- sort(c(seq(window.size), -seq(window.size)), decreasing = TRUE)
    for(i.sentence in seq(sentence)){
        word <- sentence[i.sentence]

        if(word != 0){
            for(i.context in seq(offsets)){
                offset <- offsets[i.context]
                
                i <- offset + i.sentence
                j <- word + n.vocab*(i.context - 1)
                
                if(i >= 1 && i <= n.train.words){
                    indices[(i.sentence - 1)*(2*window.size) + i.context, ] <- c(i, j)
                }
            }
        }
        setTxtProgressBar(pb, i.sentence)
    }

    indices <- indices[rowSums(indices) > 0, ]
    C <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                      x = rep(1, times = nrow(indices)),
                      dims = c(n.train.words, 2*window.size*n.vocab))
    
    ## CCAを実行
    redsvd.A <- cca.redsvd(W, C, dim.internal)

    return.list <- list()
    return.list$svd <- redsvd.A
    return.list$vocab.words <- vocab.words
    
    return(return.list)
}


most.similar <- function(query, res.eigenwords, topn = 10){
    vocab <- res.eigenwords$vocab.words
    rep.vocab <- res.eigenwords$svd$U
    
    if (!query %in% vocab){
        print(paste0("Error: `", query, "` is not in vocaburary."))
        return(FALSE)
    }

    index.query <- which(vocab == query)
    rep.query <- rep.vocab[index.query, ]
    rep.query.matrix <- matrix(rep.query, nrow=length(vocab), ncol=length(rep.query), byrow=TRUE)
    distances <- sqrt(rowSums((rep.vocab - rep.query.matrix)**2))
    names(distances) <- vocab
    
    return(sort(distances)[1:topn])
}


## Tuning parameters
min.count <- 10       # 出現回数がmin.count回以下の単語はvocabに入れない
dim.internal <- 200   # 共通空間の次元
window.size <- 2      # 前後何個の単語をcontextとするか

## Making train data
f <- file("train_all.txt", "r")
line <- readLines(con = f, -1)
close(f)

sentence.orig.full <- unlist(strsplit(tolower(line), " "))
sentence.orig <- sentence.orig.full#[1:100000]
vocab.orig <- unique(sentence.orig)

## Eigenwords
res.eigenwords <- eigenwords(sentence.orig, vocab.orig, min.count, dim.internal, window.size)

## Check vector representations
most.similar("安全", res.eigenwords)
