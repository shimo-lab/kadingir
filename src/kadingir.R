### Implementation of Eigenwords and its extensions

library(Rcpp)
library(RcppEigen)
library(foreach)
library(doParallel)

sourceCpp("kadingir_rcpp_wrapper.cpp", rebuild = TRUE)



Eigenwords <- function(path.corpus, max.vocabulary = 1000, dim.internal = 200,
                       window.size = 2, mode = "oscca", plot = FALSE) {
  
  time.start <- Sys.time()
  
  ## Making train data
  f <- file(path.corpus, "r")
  lines <- readLines(con = f, -1)
  close(f)
  
  lines.splited <- strsplit(lines, " ")
  sentence.str <- unlist(lines.splited)
  rm(lines)
  rm(lines.splited)
  
  if (!mode %in% c("oscca", "tscca")) {
    cat(paste0("mode is invalid: ", mode))
  }
  
  d.table <- table(sentence.str)
  d.table.sorted <- sort(d.table, decreasing = TRUE)
  vocab.words <- names(d.table.sorted[seq(max.vocabulary)])
  sentence <- match(sentence.str, vocab.words, nomatch = 0)  # Fill zero for out-of-vocabulary words
  rm(sentence.str)
  n.vocab <- max.vocabulary + 1  # For out-of-vocabulary word, +1
  
  if (plot) {
    plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
    abline(v = max.vocabulary)    
  }
  
  cat("\n\n")
  cat("Corpus             :", path.corpus, "\n")
  cat("Size of sentence   :", length(sentence), "\n")
  cat("dim.internal       :", dim.internal, "\n")
  cat("window.size        :", window.size, "\n")
  cat("Size of vocab      :", n.vocab, "\n")
  cat("mode               :", mode, "\n\n")

  
  sentence <- as.integer(sentence)
  
  if (mode == "oscca") {
    results.redsvd <- EigenwordsOSCCACpp(sentence, window.size, n.vocab, dim.internal, FALSE)
  } else {
    results.redsvd <- EigenwordsTSCCACpp(sentence, window.size, n.vocab, dim.internal, FALSE)
  }
  
  return.list <- list()
  return.list$svd <- results.redsvd
  return.list$vocab.words <- c("<OOV>", vocab.words)
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}


Eigendocs <- function(path.corpus, max.vocabulary = 1000, dim.internal = 200,
                      window.size = 2, plot = FALSE) {

  link_w_d <- TRUE
  link_c_d <- TRUE
  
  time.start <- Sys.time()
  
  ## Making train data
  f <- file(path.corpus, "r")
  lines <- readLines(con = f, -1)
  close(f)
  
  lines.splited <- strsplit(lines, " ")
  sentence.str <- unlist(lines.splited)
  lines.splited.lengths <- sapply(lines.splited, length)
  document.id <- rep(seq(lines.splited), lines.splited.lengths) - 1L
  
  if (plot) {
    hist(lines.splited.lengths, breaks = 100)    
  }
  rm(lines)
  
  d.table <- table(sentence.str)
  d.table.sorted <- sort(d.table, decreasing = TRUE)
  vocab.words <- names(d.table.sorted[seq(max.vocabulary)])
  sentence <- match(sentence.str, vocab.words, nomatch = 0)  # Fill zero for out-of-vocabulary words
  rm(sentence.str)
  
  if (plot) {
    plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
    abline(v = max.vocabulary)    
  }
  rm(d.table)
  rm(d.table.sorted)
  
  n.vocab <- max.vocabulary + 1  # For out-of-vocabulary word, +1
  
  cat("\n\n")
  cat("Corpus             :", path.corpus, "\n")
  cat("Size of sentence   :", length(sentence), "\n")
  cat("# of documents     :", max(document.id), "\n")
  cat("dim.internal       :", dim.internal, "\n")
  cat("window.size        :", window.size, "\n")
  cat("Size of vocab      :", n.vocab, "\n")
  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")
  
  cat("Calculate Eigendocs...\n\n")
  

  results.redsvd <- EigendocsCpp(as.integer(sentence), as.integer(document.id),
                                 window.size, n.vocab, dim.internal,
                                 link_w_d = link_w_d, link_c_d = link_c_d, FALSE)
  
  if (plot) {
    plot(results.redsvd$singular_values, log = "y", main = "Singular Values")
  }
  
  return.list <- list()
  return.list$svd <- results.redsvd
  return.list$vocab.words <- c("<OOV>", vocab.words)
  return.list$document_id <- seq(nrow(results.redsvd$document_vector))

  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}


MostSimilar <- function(U, vocab, positive = NULL, negative = NULL,
                        topn = 10, distance = "euclid", print.error = TRUE,
                        language.search = NULL, weight.vector = NULL) {
  
  rownames(U) <- vocab
  
  if (distance == "cosine") {
    if (!is.null(weight.vector)) {
      U <- U %*% diag(weight.vector)
    }
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
  
  if (!is.null(language.search)) {
    rownames.U <- row.names(U)
    search.indices <- grepl(paste0("^\\(", language.search, "\\)"), rownames.U)
    U <- U[search.indices, ]
  }
    
  if (distance == "euclid") {
    rep.query.matrix <- matrix(rep.query, nrow=nrow(U), ncol=ncol(U), byrow=TRUE)
    
    if (is.null(weight.vector)) {
      weight.vector <- rep(1, times = ncol(U))
    }
    
    distances <- sqrt(rowSums(((U - rep.query.matrix) %*% diag(weight.vector))**2))
    return(distances[order(distances)[1:topn]])
  } else if (distance == "cosine") {
    similarities <- drop(U %*% rep.query)
    return(similarities[order(-similarities)[1:topn]])
  }
}


MostSimilarDocs <- function (document.id, document_vector, titles, topn = 10) {
  
  cat("QUERY :\t\t", as.character(titles[document.id]), "\n\n")
  
  rownames(document_vector) <- titles
  document_vector <- document_vector/sqrt(rowSums(document_vector**2))
  similarities <- drop(document_vector %*% document_vector[document.id, ])
  
  most_similar <- similarities[order(similarities, decreasing = TRUE)[2:(topn+1)]]
  for (i in seq(most_similar)) {
    cat(most_similar[i], "\t", names(most_similar)[i], "\n")
  }
}


TestGoogleTasks <- function (U, vocab, path = "./../data/word2vec/questions-words.txt", n.cores = 1) {
  
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


TestWordsim353 <- function (vectors, vocab, path = "./../data/wordsim353/combined.csv") {
  
  cosine.similarity <- function (w1, w2) {
    v1 <- vectors[w1, ]
    v2 <- vectors[w2, ]
    
    (v1 %*% v2) / sqrt((v1 %*% v1) * (v2 %*% v2))
  }
  
  
  ## Calculate similarities and correlation
  rownames(vectors) <- vocab
  
  wordsims <- read.csv(path, header = 1)
  
  n.tests <- nrow(wordsims)
  similarity.eigenwords <- rep(NA, times = n.tests)
  
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
