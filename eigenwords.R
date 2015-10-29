### Implementation of Eigenwords

library(Matrix)
library(RRedsvd)
library(svd)
library(Rcpp)
library(RcppEigen)

sourceCpp("rcppeigenwords.cpp")


make.matrices <- function(sentence, window.size, n.train.words, n.vocab) {

  ## Construction of W
  indices <- cbind(seq(sentence), sentence)
  indices <- indices[indices[ , 2] > 0, ]

  W <- sparseMatrix(i = indices[ , 1], j = indices[ , 2],
                    x = rep(1, times = nrow(indices)),
                    dims = c(n.train.words, n.vocab))

  ## Construction of C
  offsets <- c(-window.size:-1, 1:window.size)
  C <- Matrix(F, nrow = n.train.words, ncol = 0)

  for (i.offset in seq(offsets)) {
    offset <- offsets[i.offset]
    indices.temp <- cbind(seq(sentence) - offset, sentence)

    # Ignore invalid indices and indices of null words
    indices.temp <- indices.temp[(indices.temp[ , 1] > 0) & (indices.temp[ , 1] <= n.train.words), ]
    indices.temp <- indices.temp[indices.temp[ , 2] > 0, ]

    C.temp <- sparseMatrix(i = indices.temp[, 1], j = indices.temp[, 2],
                           x = rep(1, times = nrow(indices.temp)),
                           dims = c(n.train.words, n.vocab))
    C <- cbind2(C, C.temp)
  }

  return(list(W = W, C = C))
}

TruncatedSVD <- function(A, k, sparse) {
  print("TruncatedSVD()...")
  
  if (sparse) {
    results.svd <- redsvd(A, k)
  } else {
    results.propack.svd <- propack.svd(as.matrix(A), neig=k)
    results.svd <- list()
    results.svd$U <- results.propack.svd$u
    results.svd$V <- results.propack.svd$v
    results.svd$D <- results.propack.svd$d
  }
  print("End of TruncatedSVD()")
  return(results.svd)
  
}

crossprod.block <- function(X, Y = NULL, only.diag = FALSE) {
  
  if (is.null(Y)) {
    Y <- X
  }
  
  ZZ <- Matrix(FALSE, nrow = 0, ncol = sum(sapply(Y, ncol)))
  for (i in seq(X)) {
    Zi <- Matrix(FALSE, nrow = ncol(X[[i]]), ncol = 0)
    for (j in seq(Y)) {
      if (only.diag && i != j) {
        Zi <- cbind2(Zi, Matrix(FALSE, nrow = ncol(X[[i]]), ncol = ncol(Y[[j]])))
      } else {
        Zi <- cbind2(Zi, crossprod(X[[i]], Y[[j]]))        
      }
    }
    ZZ <- rbind2(ZZ, Zi)
  }
  
  return(ZZ)
}

OSCCA <- function(X, Y, k, use.block.matrix) {
  ## CCA using randomized SVD
  ##  In the same way as [Dhillon+2015],
  ##  ignore off-diagonal elements of Cxx & Cyy
  ##
  ##  Arguments :
  ##    X : matrix or list of matrices
  ##    Y : matrix or list of matrices
  ##    k : number of desired singular values
  ##    use.block.matrix : Are X, Y block matrix?
  
  if (use.block.matrix) {
    Cxx <- crossprod.block(X, only.diag = TRUE)
    Cxy <- crossprod.block(X, Y)
    Cyy <- crossprod.block(Y, only.diag = TRUE)
  } else {
    Cxx <- crossprod(X)
    Cxy <- crossprod(X, Y)
    Cyy <- crossprod(Y)
  }
  
  A <- Diagonal(nrow(Cxx), diag(Cxx)^(-1/2)) %*% Cxy %*% Diagonal(nrow(Cyy), diag(Cyy)^(-1/2))
  
  cat("Calculate redsvd...")
  return(TruncatedSVD(A, k, sparse = TRUE))
  
}

TSCCA <- function(W, L, R, k) {
  redsvd.LR <- OSCCA(L, R, k)
  U <- redsvd.LR$U
  V <- redsvd.LR$V
  
  Cww <- crossprod(W)
  Css <- rbind2(
    cbind2(
      t(U) %*% crossprod(L) %*% U,
      t(U) %*% crossprod(L, R) %*% V
    ),
    cbind2(
      t(V) %*% crossprod(R, L) %*% U,
      t(V) %*% crossprod(R) %*% V
    )
  )
  Cws <- cbind2(
    crossprod(W, L) %*% U,
    crossprod(W, R) %*% V
  )
  
  A <- diag(diag(Cww)^(-1/2)) %*% Cws %*% diag(diag(Css)^(-1/2))
  
  return(TruncatedSVD(A, k, sparse = FALSE))
}


Eigenwords <- function(sentence.orig, min.count = 10,
                       dim.internal = 200, window.size = 2, mode = "oscca",
                       use.block.matrix = FALSE) {
  
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

  if (use.eigen) {
    sentence <- as.integer(sentence) - 1L
    r <- MakeMatrices(sentence, window.size, length(unique(sentence))-1)
  } else {
    r <- make.matrices(sentence, window.size, n.train.words, n.vocab)
  }
  
  cat("Size of W :")
  print(object.size(r$W), unit = "GB")
  cat("Size of C :")
  print(object.size(r$C), unit = "GB")

  ## Execute CCA
  if (mode == "oscca") { # One-step CCA
    cat("Calculate OSCCA...\n\n")
#    results.redsvd <- TruncatedSVD(A = MakeSVDMatrix(r$W, r$C), k = dim.internal, sparse = TRUE)
    results.redsvd <- RedsvdOSCCA(r$W, r$C, k = dim.internal)

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


TestGoogleTasks <- function (res.eigenwords, path) {
  ## Calcurate accuracy of Google analogy task
  queries <- read.csv(path, header = FALSE,
                      sep = " ", comment.char = ":")
  
  time.start <- Sys.time()
  results <- rep(NULL, times = nrow(queries))
  for (i in seq(nrow(queries))) {
    q <- as.character(unlist(queries[i, ]))
    res.MostSimilar <- MostSimilar(res.eigenwords, positive=c(q[[2]], q[[3]]),
                                   negative=c(q[[1]]),
                                   format="euclid", topn=10, print.error = FALSE)
    
    results[i] <- res.MostSimilar && names(res.MostSimilar)[[1]] == q[4]
  }
  print(Sys.time() - time.start)
  
  cat("accuracy = ", sum(results), "/", length(results))
}
