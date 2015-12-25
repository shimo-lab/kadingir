
library(Rcpp)
library(RcppEigen)

sourceCpp("kadingir_core.cpp", rebuild = TRUE, verbose = TRUE)


MCEigendocs <- function(paths.corpus, max.vocabulary = 1000, dim.internal = 200,
                        window.sizes = NULL, aliases = NULL, plot = FALSE,
                        link_w_d = TRUE, link_c_d = TRUE, doc_weighting = TRUE)
{  
  time.start <- Sys.time()
  
  ## Preprocess training data
  sentences <- list()
  vocab.sizes <- c()
  document.id <- list()
  vocab.words <- list()
  
  for (i in seq(paths.corpus)) {
    path.corpus <- paths.corpus[i]
    
    cat(path.corpus, "\n")
    
    f <- file(path.corpus, "r")
    lines <- readLines(con = f, -1)
    close(f)
    
    lines.splited <- strsplit(lines, " ")
    sentence.str <- unlist(lines.splited)
    lines.splited.lengths <- sapply(lines.splited, length)
    document.id[[i]] <- rep(seq(lines.splited), lines.splited.lengths) - 1L

    if (plot) {
      hist(lines.splited.lengths, breaks = 100)
    }
    
    rm(lines)
    
    d.table <- table(sentence.str)
    d.table.sorted <- sort(d.table, decreasing = TRUE)
    vocab.words[[i]] <- names(d.table.sorted[seq(max.vocabulary)])
    sentences[[i]] <- match(sentence.str, vocab.words[[i]], nomatch = 0)  # Fill zero for out-of-vocabulary words
    rm(sentence.str)

    if (plot) {
      plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
      abline(v = max.vocabulary)
    }
    
    rm(d.table)
    rm(d.table.sorted)
    
    vocab.sizes[i] <- max.vocabulary + 1  # For out-of-vocabulary word, +1
  }
  
  cat("\n\n")
  
  cat("dim.internal       :", dim.internal, "\n")
  cat("Doc weighting      :", doc_weighting, "\n")
  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")
  
  for (i in seq(paths.corpus)) {
    cat("===== Corpus #", i, " =====\n", sep="")
    cat("Alias              :", aliases[i], "\n")
    cat("Path               :", paths.corpus[i], "\n")
    cat("Size of sentence   :", length(sentences[[i]]), "\n")
    cat("# of documents     :", max(document.id[[i]]), "\n")
    cat("window.size        :", window.sizes[i], "\n")
    cat("Size of vocab      :", vocab.sizes[i], "\n\n")
  }
  
  
  cat("Calculate MCEigendocs...\n\n")
    
  corpus.concated <- as.integer(unlist(sentences))
  document.id.concated <- as.integer(unlist(document.id))
  window.sizes <- as.integer(window.sizes)
  vocab.sizes <- as.integer(vocab.sizes)
  sentence.lengths <- lengths(sentences)
  n.languages <- length(paths.corpus)
  
  results.redsvd <- MCEigendocsRedSVD(corpus.concated, document.id.concated,
                                      window.sizes, vocab.sizes, sentence.lengths,
                                      dim.internal, gamma_G = 0, gamma_H = 0,
                                      link_w_d = link_w_d, link_c_d = link_c_d, doc_weighting = doc_weighting)
  
  return.list <- list()
  return.list$svd <- results.redsvd


  return.list$vocab.words <- list()
  for (i in seq(n.languages)) {
    return.list$vocab.words <- c(return.list$vocab.words, list(c("<OOV>", vocab.words[[i]])))
  }

  return.list$document_id <- seq(nrow(results.redsvd$document_vector))
  return.list$n.languages <- length(sentence.lengths)
  return.list$sentence.lengths <- sentence.lengths
  return.list$vocab.sizes <- vocab.sizes
  
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}
