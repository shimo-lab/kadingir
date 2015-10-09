Eigenwords
================

# Installation of RRedsvd

    install.packages("devtools")
    library(devtools)
    install_github("xiangze/RRedsvd")

# memo
## "Zipfian nature" の確認
`plot(sort(table(sentence.orig), decreasing = TRUE), log="y")`

## How to get data files
`scp poomo:/data/oshikiri/eigenwords/* ./`

# Reference
* Dhillon, P. S., Foster, D. P., & Ungar, L. H. (2015). Eigenwords : Spectral Word Embeddings, 16. Retrieved from http://www.pdhillon.com/dhillon15a.pdf
* questions-words.txt <https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt>

