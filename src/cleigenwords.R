
library(Rcpp)
library(RcppEigen)

sourceCpp("kadingir_core.cpp", rebuild = TRUE)


CLEigenwords <- function(paths.corpus, sizes.vocabulary, dim.common,
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
  
  n.languages <- length(aliases.languages)
  
  for (i in seq(n.languages)) {
    language <- aliases.languages[i]
    path.corpus <- paths.corpus[[language]]
    
    # Preprocess parallel corpus
    if (exists("parallel", where = path.corpus)) {
      cat(path.corpus[["parallel"]], "\n")
      f.parallel <- file(path.corpus[["parallel"]], "r")
      lines.parallel <- readLines(con = f.parallel, -1)
      close(f.parallel)
    } else {
      stop(paste0("Error: No parallel corpus of language ", language, "."))
    }

    # Preprocess monolingual corpus
    if (exists("monolingual", where = path.corpus)) {
      cat(path.corpus[["monolingual"]], "\n")
      f.monolingual <- file(path.corpus[["monolingual"]], "r")
      lines.monolingual <- readLines(con = f.monolingual, -1)
      close(f.monolingual)
    } else {
      lines.monolingual <- ""
    }
    
    lines.parallel.splited <- strsplit(lines.parallel, " ")
    lines.monolingual.splited <- strsplit(lines.monolingual, " ")
    sentence.str <- unlist(c(lines.parallel.splited, lines.monolingual.splited))
    lengths.lines.parallel.splited <- sapply(lines.parallel.splited, length)
    length.lines.monolingual.splited <- sum(sapply(lines.monolingual.splited, length))
    
    document.id.parallel <- rep(seq(lines.parallel.splited) - 1L, lengths.lines.parallel.splited)
    document.id.monolingual <- rep(-1L, length.lines.monolingual.splited)
    document.id[[i]] <- c(document.id.parallel, document.id.monolingual)

    if (plot) {
      hist(lengths.lines.parallel.splited, breaks = 100)
    }
    
    rm(lines.monolingual, lines.parallel, length.lines.monolingual.splited, lengths.lines.parallel.splited)
    
    d.table <- table(sentence.str)
    d.table.sorted <- sort(d.table, decreasing = TRUE)
    vocab.words[[i]] <- names(d.table.sorted[seq(sizes.vocabulary[i] - 1)])  # For out-of-vocabulary word, -1
    sentences[[i]] <- match(sentence.str, vocab.words[[i]], nomatch = 0)  # Fill zero for out-of-vocabulary words
    min.counts[[i]] <- d.table.sorted[[sizes.vocabulary[i] - 1]]

    if (plot) {
      plot(d.table.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
      abline(v = sizes.vocabulary[i])
    }
    
    rm(sentence.str, d.table, d.table.sorted)
  }
  
  cat("\n\n")
  
  cat("Dim of common space:", dim.common, "\n")
  cat("Weight by TF?      :", weighting_tf, "\n")

  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")
  
  for (i in seq(n.languages)) {
    language <- aliases.languages[i]
    
    cat("===== Corpus #", i, " =====\n", sep="")
    cat("Alias              :", language, "\n")
    cat("Monolingual corpus :", paths.corpus[[language]][["monolingual"]], "\n")
    cat("Parallel corpus    :", paths.corpus[[language]][["parallel"]], "\n")
    cat("Size of sentence   :", length(sentences[[i]]), "\n")
    cat("Coverage           :", 100 * mean(sentences[[i]] > 0), "\n")
    cat("# of documents     :", max(document.id[[i]]) + 1L, "\n")
    cat("size.window        :", sizes.window[i], "\n")
    cat("Size of vocabulary :", sizes.vocabulary[i], "\n")
    cat("Weight (vs doc)    :", weight.vsdoc[i], "\n")
    cat("min count          :", min.counts[[i]], "\n")
    cat("% of docid >= 0    :", 100 * mean(document.id[[i]] >= 0), "\n")
    
    cat("\n")
  }

  cat("Calculate CLEigenwords...\n\n")
  
  corpus.concated <- as.integer(unlist(sentences))
  document.id.concated <- as.integer(unlist(document.id))
  sizes.window <- as.integer(sizes.window)
  sizes.vocabulary <- as.integer(sizes.vocabulary)
  lengths.sentence <- lengths(sentences)
  n.languages <- length(paths.corpus)
  
  vocab.words.concated <- list()
  for (i in seq(n.languages)) {
    vocab.words.concated <- c(vocab.words.concated, list(c("<OOV>", vocab.words[[i]])))
  }
  
  results.redsvd <- CLEigenwordsCpp(corpus.concated, document.id.concated,
                                    sizes.window, sizes.vocabulary, lengths.sentence,
                                    dim.common, gamma_G = 0, gamma_H = 0,
                                    link_w_d = link_w_d, link_c_d = link_c_d,
                                    weighting_tf = weighting_tf,
                                    weight_vsdoc = weight.vsdoc,
                                    debug = FALSE)
  
  return.list <- list()
  return.list$svd <- results.redsvd

  if (plot) {
    plot(results.redsvd$singular_values, log = "y", main = "Singular Values")
  }

  return.list$corpus.concated <- corpus.concated
  return.list$document.id.concated <- document.id.concated
  return.list$vocab.words <- vocab.words.concated
  return.list$document_id <- seq(nrow(results.redsvd$document_vector))
  return.list$n.languages <- length(lengths.sentence)
  return.list$lengths.sentence <- lengths.sentence
  return.list$sizes.vocabulary <- sizes.vocabulary
  
  
  diff.time <- Sys.time() - time.start
  print(diff.time)
  
  return(return.list)
}
