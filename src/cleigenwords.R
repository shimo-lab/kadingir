
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
  id.wordtype <- list()
  id.document <- list()
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
    str.wordtype <- unlist(c(lines.parallel.splited, lines.monolingual.splited))
    lengths.lines.parallel.splited <- sapply(lines.parallel.splited, length)
    length.lines.monolingual.splited <- sum(sapply(lines.monolingual.splited, length))
    
    id.document.parallel <- rep(seq(lines.parallel.splited) - 1L, lengths.lines.parallel.splited)
    id.document.monolingual <- rep(-1L, length.lines.monolingual.splited)
    id.document[[i]] <- c(id.document.parallel, id.document.monolingual)

    if (plot) {
      hist(lengths.lines.parallel.splited, breaks = 100)
    }
    
    rm(lines.monolingual, lines.parallel, length.lines.monolingual.splited, lengths.lines.parallel.splited)
    
    table.wordtype <- table(str.wordtype)
    table.wordtype.sorted <- sort(table.wordtype, decreasing = TRUE)
    vocab.words[[i]] <- names(table.wordtype.sorted[seq(sizes.vocabulary[i] - 1)])  # For out-of-vocabulary word, -1
    id.wordtype[[i]] <- match(str.wordtype, vocab.words[[i]], nomatch = 0)  # Fill zero for out-of-vocabulary words
    min.counts[[i]] <- table.wordtype.sorted[[sizes.vocabulary[i] - 1]]

    if (plot) {
      plot(table.wordtype.sorted, log="xy", col=rgb(0, 0, 0, 0.1))
      abline(v = sizes.vocabulary[i])
    }
    
    rm(str.wordtype, table.wordtype, table.wordtype.sorted)
  }
  
  cat("\n\n")
  
  cat("Dim of common space:", dim.common, "\n")
  cat("Weight by TF?      :", weighting_tf, "\n")
  cat("Link: W - D        :", link_w_d, "\n")
  cat("Link: C - D        :", link_c_d, "\n\n")

  # Print informations of each languages
  for (i in seq(n.languages)) {
    language <- aliases.languages[i]
    
    cat("===== Corpus #", i, " =====\n", sep="")
    cat("Alias              :", language, "\n")
    cat("Monolingual corpus :", paths.corpus[[language]][["monolingual"]], "\n")
    cat("Parallel corpus    :", paths.corpus[[language]][["parallel"]], "\n")
    cat("# of tokens        :", length(id.wordtype[[i]]), "\n")
    cat("Coverage           :", 100 * mean(id.wordtype[[i]] > 0), "\n")
    cat("# of documents     :", max(id.document[[i]]) + 1L, "\n")
    cat("size.window        :", sizes.window[i], "\n")
    cat("Size of vocabulary :", sizes.vocabulary[i], "\n")
    cat("Weight (vs doc)    :", weight.vsdoc[i], "\n")
    cat("min count          :", min.counts[[i]], "\n")
    cat("% of docid >= 0    :", 100 * mean(id.document[[i]] >= 0), "\n")
    cat("\n")
  }

  id.wordtype.concated <- as.integer(unlist(id.wordtype))
  id.document.concated <- as.integer(unlist(id.document))
  sizes.window <- as.integer(sizes.window)
  sizes.vocabulary <- as.integer(sizes.vocabulary)
  lengths.corpus <- lengths(id.wordtype)
  n.languages <- length(paths.corpus)
  
  vocab.words.concated <- list()
  for (i in seq(n.languages)) {
    vocab.words.concated <- c(vocab.words.concated, list(c("<OOV>", vocab.words[[i]])))
  }

  cat("Calculate CLEigenwords...\n\n")
  results.redsvd <- CLEigenwordsCpp(id.wordtype.concated, id.document.concated,
                                    sizes.window, sizes.vocabulary, lengths.corpus,
                                    dim.common,
                                    link_w_d = link_w_d, link_c_d = link_c_d,
                                    weighting_tf = weighting_tf,
                                    weight_vsdoc = weight.vsdoc,
                                    debug = FALSE)

  if (plot) {
    plot(results.redsvd$singular_values, log = "y", main = "Singular Values")
  }

  return.list <- list()
  return.list$svd <- results.redsvd
  return.list$id.wordtype.concated <- id.wordtype.concated
  return.list$id.document.concated <- id.document.concated
  return.list$vocab.words <- vocab.words.concated
  return.list$document_id <- seq(nrow(results.redsvd$document_vector))
  return.list$n.languages <- length(lengths.corpus)
  return.list$lengths.corpus <- lengths.corpus
  return.list$sizes.vocabulary <- sizes.vocabulary

  diff.time <- Sys.time() - time.start
  print(diff.time)

  return(return.list)
}
