## Implementation of Eigenwords

library(Matrix)
library(RRedsvd)
library(tcltk)
library(svd)


## CCA using randomized SVD
##  In the same way as [Dhillon+2015], ignore off-diagonal elements of Cxx & Cyy
##
##  Arguments :
##    X : matrix
##    Y : matrix
##    k : number of desired singular values
##    sparse : Use redsvd or propack.svd?
cca.eigenwords <- function(X, Y, k, sparse = TRUE){
    Cxx <- t(X) %*% X
    Cxy <- t(X) %*% Y
    Cyy <- t(Y) %*% Y
    
    A <- Diagonal(nrow(Cxx), diag(Cxx)^(-1/2)) %*% Cxy %*% Diagonal(nrow(Cyy), diag(Cyy)^(-1/2))

    if (sparse) {
        results.svd <- redsvd(A, k)
    } else {
        results.propack.svd <- propack.svd(as.matrix(A), neig=k)
        results.svd <- list()
        results.svd$U <- results.propack.svd$u
        results.svd$V <- results.propack.svd$v
        results.svd$D <- results.propack.svd$d
    }

    return(results.svd)
}


eigenwords <- function(sentence.orig, vocab.orig, min.count = 10,
                       dim.internal = 200, window.size = 2, mode = "oscca"){

    if (!mode %in% c("oscca", "tscca")){
        cat(paste0("mode is invalid: ", mode))
    }
    
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
    cat("\nConstructing W\n")
    pb <- txtProgressBar(min = 1, max = length(sentence), style = 3)
    
    indices <- matrix(0, nrow = length(sentence), ncol = 2)
    for(i.sentence in seq(sentence)){    
        word <- sentence[i.sentence]

        if(word != 0){
            indices[i.sentence, ] <- c(i.sentence, word)
        }

        setTxtProgressBar(pb, i.sentence)
    }
    cat("\n\n")
    
    indices <- indices[rowSums(indices) > 0, ]
    W <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                      x = rep(1, times = nrow(indices)),
                      dims = c(n.train.words, n.vocab))
    
    ## C を構成
    cat("\nConstructing C\n")
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
    if (mode == "oscca") { # One-step CCA
        cat("Calculate OSCCA...")
        results.redsvd <- cca.eigenwords(W, C, dim.internal)
    } else if (mode == "tscca") { # Two-Step CCA
        cat("Calculate TSCCA...")
        L <- C[ , 1:(window.size*n.vocab)]
        R <- C[ , (window.size*n.vocab+1):(2*window.size*n.vocab)]
        redsvd.LR <- cca.eigenwords(L, R, dim.internal)

        S <- cbind(L %*% redsvd.LR$U, R %*% redsvd.LR$V)
        results.redsvd <- cca.eigenwords(W, S, dim.internal, sparse = FALSE)
    }

    return.list <- list()
    return.list$svd <- results.redsvd
    return.list$vocab.words <- vocab.words
    
    return(return.list)
}


most.similar <- function(res.eigenwords, positive = NULL, negative = NULL, topn = 10){
    vocab <- res.eigenwords$vocab.words
    rep.vocab <- res.eigenwords$svd$U

    queries.info <- list(list(positive, 1), list(negative, -1))
    rep.query <- rep(0, times=ncol(rep.vocab))

    for (q in queries.info) {
        queries <- q[[1]]
        pm <- q[[2]]
        
        if (!is.null(queries)) {
            for (query in queries) {
                
                if (!query %in% vocab) {
                    print(paste0("Error: `", query, "` is not in vocaburary."))
                    return(FALSE)
                }
                
                index.query <- which(vocab == query)
                rep.query <- rep.query + pm * rep.vocab[index.query, ]
            }
        }
    }

    rep.query.matrix <- matrix(rep.query, nrow=length(vocab), ncol=length(rep.query), byrow=TRUE)
    distances <- sqrt(rowSums((rep.vocab - rep.query.matrix)**2))
    names(distances) <- vocab
    
    return(sort(distances)[1:topn])
}
