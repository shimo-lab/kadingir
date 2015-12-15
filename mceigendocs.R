MCEigendocs <- function(paths.corpus, max.vocabulary = 1000, dim.internal = 200,
                        window.sizes = NULL, aliases = NULL) {
  
  link_w_d <- TRUE
  link_c_d <- TRUE
  
  time.start <- Sys.time()
  
  ## Making train data
  sentences <- list()
  n.vocab <- c()
  document.id <- list()
  
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
    
    hist(lines.splited.lengths, breaks = 100)
    rm(lines)
    
    d.table <- table(sentence.str)
    d.table.sorted <- sort(d.table, decreasing = TRUE)
    vocab.words <- names(d.table.sorted[seq(max.vocabulary)])
    sentences[[i]] <- match(sentence.str, vocab.words, nomatch = 0)  # Fill zero for out-of-vocabulary words
    rm(sentence.str)
    
    #     plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
    #     abline(v = max.vocabulary)
    rm(d.table)
    rm(d.table.sorted)
    
    n.vocab[i] <- max.vocabulary + 1  # For out-of-vocabulary word, +1
  }
  
  cat("\n\n")
  
  cat("dim.internal       :", dim.internal, "\n")
  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")
  
  for (i in seq(paths.corpus)) {
    cat("===== Corpus #", i, " =====\n", sep="")
    cat("Alias              :", aliases[i], "\n")
    cat("Path               :", paths.corpus[i], "\n")
    cat("Size of sentence   :", length(sentences[[i]]), "\n")
    cat("# of documents     :", max(document.id[[i]]), "\n")
    cat("window.size        :", window.sizes[i], "\n")
    cat("Size of vocab      :", n.vocab[i], "\n\n")
  }
  
  
  cat("Calculate MCEigendocs...\n\n")
  
  browser()
  
  # とりあえず2言語で．n言語にするの難しそう...
  results.redsvd <- MCEigendocsRedSVD(as.integer(sentences[1]), as.integer(document.id[1]), window.sizes[1], n.vocab[1],
                                      as.integer(sentences[2]), as.integer(document.id[2]), window.sizes[2], n.vocab[2],
                                      dim.internal,
                                      gamma_G = 0, gamma_H = 0, link_w_d = link_w_d, link_c_d = link_c_d)
  
  
  return.list <- list()
  return.list$svd <- results.redsvd
  return.list$vocab.words <- c("<OOV>", vocab.words)
  return.list$document_id <- seq(nrow(results.redsvd$document_vector))
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}
