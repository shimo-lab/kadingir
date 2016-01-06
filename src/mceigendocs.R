
library(Rcpp)
library(RcppEigen)

sourceCpp("kadingir_core.cpp", rebuild = TRUE)


MCEigendocs <- function(paths.corpus, sizes.vocabulary, dim.common,
                        sizes.window, aliases.languages, weight.vsdoc,
                        plot = FALSE,
                        link_w_d = TRUE, link_c_d = TRUE,
                        weighting_tf = FALSE)
{
  time.start <- Sys.time()
  
  ## Preprocess training data
  sentences <- list()
  document.id <- list()
  vocab.words <- list()
  min.counts <- list()
  
  for (i in seq(paths.corpus)) {
    path.corpus <- paths.corpus[i]
    
    cat(path.corpus, "\n")
    
    f <- file(path.corpus, "r")
    lines <- readLines(con = f, -1)
    close(f)
    
    lines.splited <- strsplit(lines, " ")
    sentence.str <- unlist(lines.splited)
    lengths.lines.splited <- sapply(lines.splited, length)
    document.id[[i]] <- rep(seq(lines.splited), lengths.lines.splited) - 1L

    if (plot) {
      hist(lengths.lines.splited, breaks = 100)
    }
    
    rm(lines)
    
    d.table <- table(sentence.str)
    d.table.sorted <- sort(d.table, decreasing = TRUE)
    vocab.words[[i]] <- names(d.table.sorted[seq(sizes.vocabulary[i] - 1)])  # For out-of-vocabulary word, -1
    sentences[[i]] <- match(sentence.str, vocab.words[[i]], nomatch = 0)  # Fill zero for out-of-vocabulary words
    min.counts[[i]] <- d.table.sorted[[sizes.vocabulary[i] - 1]]
    rm(sentence.str)

    if (plot) {
      plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
      abline(v = sizes.vocabulary[i])
    }
    
    rm(d.table)
    rm(d.table.sorted)
  }
  
  cat("\n\n")
  
  cat("Dim of common space:", dim.common, "\n")
  cat("Weight by TF?      :", weighting_tf, "\n")
  
  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")
  
  for (i in seq(paths.corpus)) {
    cat("===== Corpus #", i, " =====\n", sep="")
    cat("Alias              :", aliases.languages[i], "\n")
    cat("Path               :", paths.corpus[i], "\n")
    cat("Size of sentence   :", length(sentences[[i]]), "\n")
    cat("# of documents     :", max(document.id[[i]]), "\n")
    cat("window.size        :", sizes.window[i], "\n")
    cat("Size of vocabulary :", sizes.vocabulary[i], "\n")
    cat("Weight (vs doc)    :", weight.vsdoc[i], "\n")
    cat("min count          :", min.counts[[i]], "\n")
    
    cat("\n")
  }
  
  
  cat("Calculate MCEigendocs...\n\n")
    
  corpus.concated <- as.integer(unlist(sentences))
  document.id.concated <- as.integer(unlist(document.id))
  sizes.window <- as.integer(sizes.window)
  sizes.vocabulary <- as.integer(sizes.vocabulary)
  lengths.sentence <- lengths(sentences)
  n.languages <- length(paths.corpus)
  
  results.redsvd <- MCEigendocsRedSVD(corpus.concated, document.id.concated,
                                      sizes.window, sizes.vocabulary, lengths.sentence,
                                      dim.common, gamma_G = 0, gamma_H = 0,
                                      link_w_d = link_w_d, link_c_d = link_c_d,
                                      weighting_tf = weighting_tf,
                                      weight_vsdoc = weight.vsdoc)
  
  return.list <- list()
  return.list$svd <- results.redsvd


  return.list$vocab.words <- list()
  for (i in seq(n.languages)) {
    return.list$vocab.words <- c(return.list$vocab.words, list(c("<OOV>", vocab.words[[i]])))
  }

  return.list$document_id <- seq(nrow(results.redsvd$document_vector))
  return.list$n.languages <- length(lengths.sentence)
  return.list$lengths.sentence <- lengths.sentence
  return.list$sizes.vocabulary <- sizes.vocabulary
  
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}
