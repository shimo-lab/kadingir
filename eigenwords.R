### Implementation of Eigenwords

library(Matrix)
library(RRedsvd)
library(svd)
library(Rcpp)
library(RcppEigen)
library(foreach)
library(doParallel)

sourceCpp("rcppeigenwords.cpp", rebuild = TRUE, verbose = TRUE)


make.matrices <- function(sentence, window.size, skip.null.words) {

  sentence <- sentence + 1L
  n.train.words <- length(sentence)
  n.vocab <- max(sentence)
  
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

  if (skip.null.words) {
    W <- W[indices[ , 1], ]
    C <- C[indices[ , 1], ]
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


OSCCA <- function(X, Y, k) {
  ## CCA using randomized SVD
  ##  In the same way as [Dhillon+2015],
  ##  ignore off-diagonal elements of Cxx & Cyy
  ##
  ##  Arguments :
  ##    X : matrix or list of matrices
  ##    Y : matrix or list of matrices
  ##    k : number of desired singular values
  
  Cxx <- crossprod(X)
  Cxy <- crossprod(X, Y)
  Cyy <- crossprod(Y)
  Cxx.h <- Diagonal(nrow(Cxx), diag(Cxx)^(-1/2))
  
  A <- Cxx.h %*% Cxy %*% Diagonal(nrow(Cyy), diag(Cyy)^(-1/2))
  
  cat("Calculate redsvd...")
  return.list <- TruncatedSVD(A, k, sparse = TRUE)
  return.list$word_vector <- Cxx.h %*% return.list$U

  return(return.list)
}


TSCCA <- function(W, C, k) {
  L <- C[ , 1:(ncol(C)/2)]
  R <- C[ , (ncol(C)/2 + 1):ncol(C)]

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

  Cxx.h <- sqrt(diag(diag(Cww)^(-1/2)))
  A <- Cxx.h %*% sqrt(Cws) %*% sqrt(diag(diag(Css)^(-1/2)))

  return.list <- TruncatedSVD(A, k, sparse = FALSE)
  return.list$word_vector <- Cxx.h %*% return.list$U

  return(return.list)
}


Eigenwords <- function(path.corpus, n.vocabulary = 1000, dim.internal = 200,
                       window.size = 2, mode = "oscca", use.eigen = TRUE) {
  
  time.start <- Sys.time()
  
  ## Making train data
  f <- file(path.corpus, "r")
  line <- readLines(con = f, -1)
  close(f)
  
  sentence.orig <- unlist(strsplit(line, " "))
  rm(line)
  
  if (!mode %in% c("oscca", "tscca")) {
    cat(paste0("mode is invalid: ", mode))
  }
  
  if (n.vocabulary > 0) {
    d.table <- table(sentence.orig)
    vocab.words <- names(sort(d.table, decreasing = TRUE)[seq(n.vocabulary)])
  } else {
    vocab.words <- unique(sentence)
  }
  
  sentence <- match(sentence.orig, vocab.words, nomatch = 0)
  n.vocab <- length(vocab.words) + 1  # For null word, +1
  n.corpus <- length(sentence)
  
  cat("\n\n")
  cat("Corpus             :", path.corpus, "\n")
  cat("Size of sentence   :", n.corpus, "\n")
  cat("dim.internal       :", dim.internal, "\n")
  cat("window.size        :", window.size, "\n")
  cat("Size of vocab      :", n.vocab, "\n")
  cat("mode               :", mode, "\n\n")
  

  if (use.eigen) {
    sentence <- as.integer(sentence)
    results.redsvd <- EigenwordsRedSVD(sentence, window.size, n.vocab, dim.internal, mode_oscca = (mode == "oscca"))
    
  } else {
    r <- make.matrices(sentence, window.size, skip.null.words = TRUE)

    cat("Size of W :")
    print(object.size(r$W), unit = "GB")
    cat("Size of C :")
    print(object.size(r$C), unit = "GB")
    
    ## Execute CCA
    if (mode == "oscca") { # One-step CCA
      cat("Calculate OSCCA...\n\n")
      results.redsvd <- OSCCA(r$W, r$C, dim.internal)
      
    } else if (mode == "tscca") { # Two-Step CCA
      cat("Calculate TSCCA...\n\n")
      results.redsvd <- TSCCA(r$W, r$C, dim.internal)
    }
    
  }
  
  return.list <- list()
  return.list$svd <- results.redsvd
  return.list$vocab.words <- c("<OOV>", vocab.words)
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}


MostSimilar <- function(U, vocab, positive = NULL, negative = NULL,
                        topn = 10, distance = "euclid", print.error = TRUE) {
  
  rownames(U) <- vocab
  
  if (distance == "cosine") {
    U <- U/sqrt(rowSums(U**2))
  }
  
  queries.info <- list(list(positive, 1), list(negative, -1))
  rep.query <- rep(0, times=ncol(U))
  
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
        
        rep.query <- rep.query + pm * U[query, ]
      }
    }
  }
  
  if (distance == "cosine") {
    rep.query <- rep.query/sqrt(sum(rep.query**2))
  }
  
  # Ignore query words
  query.words <- c(positive, negative)
  index.vocab.reduced <- which(!vocab %in% query.words)
  U <- U[index.vocab.reduced, ]
  vocab <- vocab[index.vocab.reduced]
    
  if (distance == "euclid") {
    rep.query.matrix <- matrix(rep.query, nrow=nrow(U), ncol=ncol(U), byrow=TRUE)
    distances <- sqrt(rowSums((U - rep.query.matrix)**2))
    return(distances[order(distances)[1:topn]])
    
  } else if (distance == "cosine") {
    similarities <- drop(U %*% rep.query)
    return(similarities[order(-similarities)[1:topn]])
  }
}


TestGoogleTasks <- function (U, vocab, path = "test/questions-words.txt", n.cores = 1) {
  
  time.start <- Sys.time()
  
  ## Calcurate accuracy of Google analogy task
  queries <- read.csv(path, header = FALSE, sep = " ", comment.char = ":")
  n.tasks <- nrow(queries)
  queries <- as.character(unlist(t(queries)))
  
  rownames(U) <- vocab
  U <- U/sqrt(rowSums(U**2))

  cl <- makeCluster(n.cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl)) 
  
  results <- foreach (i = seq(n.tasks), .combine = c) %dopar% {
    q <- queries[(4*(i-1) + 1):(4*i)]
    #q <- tolower(q)
    
    if (all(q %in% vocab)) {
      q3 <- U[q[2], ] - U[q[1], ] + U[q[3], ]
      
      similarities <- drop(U %*% q3)[!vocab %in% q[1:3]]
      most.similar.word <- names(which.max(similarities))
      
      tolower(most.similar.word) == tolower(q[4])
    } else {
      NULL
    }
  }
  print(Sys.time() - time.start)
  
  cat("accuracy = ", sum(results), "/", length(results), "=", mean(results), "\n\n")
}


TestWordsim353 <- function (vectors, vocab, path = "test/combined.csv") {
  
  cosine.similarity <- function (w1, w2) {
    v1 <- vectors[w1, ]
    v2 <- vectors[w2, ]
    
    (v1 %*% v2) / sqrt((v1 %*% v1) * (v2 %*% v2))
  }
  
  
  ## Calculate similarities and correlation
  rownames(vectors) <- vocab
  
  wordsims <- read.csv(path, header = 1)
  
  n.tests <- nrow(wordsims)
  similarity.eigenwords <- rep(NULL, times = n.tests)
  
  for(i in seq(n.tests)) {
    w1 <- as.character(wordsims[i, 1])
    w2 <- as.character(wordsims[i, 2])
    
    if (w1 %in% vocab && w2 %in% vocab) {
      similarity.eigenwords[i] <- cosine.similarity(w1, w2)
    }
  }
  similarity.human <- wordsims[ , 3]
  
  percent.used <- mean(!is.na(similarity.eigenwords))
  
  similarity.human <- similarity.human[!is.na(similarity.eigenwords)]
  similarity.eigenwords <- similarity.eigenwords[!is.na(similarity.eigenwords)]
  
  spearman.cor <- cor(similarity.human, similarity.eigenwords, method = "spearman")
  
  cat("Spearman cor.   = ", spearman.cor, "\n")
  cat("% of used pairs = ", percent.used, "\n")
  
  plot(similarity.human, similarity.eigenwords)
}
