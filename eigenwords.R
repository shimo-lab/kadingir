### Implementation of Eigenwords

library(Matrix)
library(RRedsvd)
library(svd)


TruncatedSVD <- function(A, k, sparse) {

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
OSCCA <- function(X, Y, k) {
    ## CCA using randomized SVD
    ##  In the same way as [Dhillon+2015],
    ##  ignore off-diagonal elements of Cxx & Cyy
    ##
    ##  Arguments :
    ##    X : matrix
    ##    Y : matrix
    ##    k : number of desired singular values

    Cxx <- t(X) %*% X
    Cxy <- t(X) %*% Y
    Cyy <- t(Y) %*% Y
    
    A <- Diagonal(nrow(Cxx), diag(Cxx)^(-1/2)) %*% Cxy %*% Diagonal(nrow(Cyy), diag(Cyy)^(-1/2))

    return(TruncatedSVD(A, k, sparse = TRUE))

}

TSCCA <- function(W, L, R, k) {
        redsvd.LR <- OSCCA(L, R, k)
        U <- redsvd.LR$U
        V <- redsvd.LR$V
        
        Cww <- t(W) %*% W
        Css <- rbind2(
            cbind2(
                t(U) %*% (t(L) %*% L) %*% U,
                t(U) %*% (t(L) %*% R) %*% V
            ),
            cbind2(
                t(V) %*% (t(R) %*% L) %*% U,
                t(V) %*% (t(R) %*% R) %*% V
            )
        )
        Cws <- cbind2(
            (t(W) %*% L) %*% U,
            (t(W) %*% R) %*% V
        )

        A <- diag(diag(Cww)^(-1/2)) %*% Cws %*% diag(diag(Css)^(-1/2))

        return(TruncatedSVD(A, k, sparse = FALSE))
}


Eigenwords <- function(sentence.orig, min.count = 10,
                       dim.internal = 200, window.size = 2, mode = "oscca") {

    time.start <- Sys.time()

    if (!mode %in% c("oscca", "tscca")) {
        cat(paste0("mode is invalid: ", mode))
    }
    
    if (min.count > 0) {
        d.table <- table(sentence.orig)
        vocab.words <- names(d.table[d.table >= min.count])
    } else {
        vocab.words <- unique(sentence)
    }

    cat("\n\n")
    cat("Size of sentence   :", length(sentence.orig), "\n")
    cat("dim.internal       :", dim.internal, "\n")
    cat("min.count          :", min.count, "\n")
    cat("window.size        :", window.size, "\n")
    cat("Size of vocab      :", length(vocab.words), "\n")
    cat("mode               :", mode, "\n\n")

    sentence <- match(sentence.orig, vocab.words, nomatch = 0)
    n.vocab <- length(vocab.words)
    n.train.words <- length(sentence)


    ## Calculate Eigenwords
    ##  実行速度の観点から，1が立つ要素のインデックスをすべて生成し，
    ##  sparseMatrix関数を使ってまとめて行列 W, C を生成している．
    
    ## Construction of W
    indices <- cbind(seq(sentence), sentence)
    indices <- indices[indices[ , 2] > 0, ]
    
    W <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                      x = rep(1, times = nrow(indices)),
                      dims = c(n.train.words, n.vocab))
    
    ## Construction of C
    offsets <- c(-window.size:-1, 1:window.size)
    indices <- c()
    for (i.offset in seq(offsets)) {
        offset <- offsets[i.offset]
        indices.temp <- cbind(seq(sentence) - offset,
                              sentence + n.vocab * (i.offset - 1))
        indices <- rbind(indices, indices.temp)
    }

    indices <- indices[(indices[ , 1] > 0) & (indices[ , 1] <= n.train.words), ]
    indices <- indices[indices[ , 2] > 0, ]

    C <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                      x = rep(1, times = nrow(indices)),
                      dims = c(n.train.words, 2*window.size*n.vocab))

    cat("\nSize of W :", format(object.size(W), unit = "GB"))
    cat("\nSize of C :", format(object.size(C), unit = "GB"))
    cat("\n\n")

    ## Execute CCA
    if (mode == "oscca") { # One-step CCA
        cat("Calculate OSCCA...\n\n")
        
        results.redsvd <- OSCCA(W, C, dim.internal)
    } else if (mode == "tscca") { # Two-Step CCA
        cat("Calculate TSCCA...\n\n")
        
        L <- C[ , 1:(window.size*n.vocab)]
        R <- C[ , (window.size*n.vocab+1):(2*window.size*n.vocab)]
        results.redsvd <- TSCCA(W, L, R, dim.internal)
    }

    return.list <- list()
    return.list$svd <- results.redsvd
    return.list$vocab.words <- vocab.words

    diff.time <- Sys.time() - time.start
    print(diff.time)
    
    return(return.list)
}


MostSimilar <- function(res.eigenwords, positive = NULL, negative = NULL,
                        topn = 10, normalize = FALSE, format = "euclid",
                        ignore.query.words = TRUE, print.error = TRUE) {
    vocab <- res.eigenwords$vocab.words
    rep.vocab <- res.eigenwords$svd$U

    if (normalize) {
        rep.vocab <- rep.vocab/sqrt(rowSums(rep.vocab**2))
    }

    queries.info <- list(list(positive, 1), list(negative, -1))
    rep.query <- rep(0, times=ncol(rep.vocab))

    for (q in queries.info) {
        queries <- q[[1]]
        pm <- q[[2]]
        
        if (!is.null(queries)) {
            for (query in queries) {
                
                if (!query %in% vocab) {
                    if (print.error) {
                        print(paste0("Error: `", query, "` is not in vocaburary."))
                    }
                    return(FALSE)
                }
                
                index.query <- which(vocab == query)
                rep.query <- rep.query + pm * rep.vocab[index.query, ]
            }
        }
    }

    if (normalize || format == "cosine") {
        rep.query <- rep.query/sqrt(sum(rep.query**2))
    }

    if (ignore.query.words) {
        query.words <- c(positive, negative)
        index.vocab.reduced <- which(!vocab %in% query.words)
        rep.vocab <- rep.vocab[index.vocab.reduced, ]
        vocab <- vocab[index.vocab.reduced]
    }

    if (format == "euclid") {
        rep.query.matrix <- matrix(rep.query, nrow=nrow(rep.vocab), ncol=ncol(rep.vocab), byrow=TRUE)
        distances <- sqrt(rowSums((rep.vocab - rep.query.matrix)**2))
        names(distances) <- vocab

        return(sort(distances)[1:topn])
        
    } else if (format == "cosine") {
        distances <- rep.vocab %*% rep.query
        names(distances) <- vocab

        return(sort(distances, decreasing = TRUE)[1:topn])

    }
}
