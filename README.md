Eigenwords
================

# Contents

* redsvd.hpp : [ntessore/redsvd-h](https://github.com/ntessore/redsvd-h)
* test/questions-words.txt : <https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt>

# Dependency

* Matrix
* RRedsvd
* svd
* Rcpp
* RcppEigen
* foreach
* doParallel


## Installation of RRedsvd

    install.packages("devtools")
    library(devtools)
    install_github("xiangze/RRedsvd")

# memo
## "Zipfian nature" の確認
* `plot(sort(table(sentence.orig), decreasing = TRUE), log="y")`
* `plot(sort(colMeans(W), decreasing = TRUE), log = "y", cex.axis = 3)`


# Reference
* Dhillon, P. S., Foster, D. P., & Ungar, L. H. (2015). Eigenwords : Spectral Word Embeddings, 16. Retrieved from http://www.pdhillon.com/dhillon15a.pdf
* questions-words.txt <https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt>

