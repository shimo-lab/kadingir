kadingir
=====================================================

[![Build Status](https://travis-ci.org/shimo-lab/kadingir.svg?branch=develop)](https://travis-ci.org/shimo-lab/kadingir)


This is an open source implementation of

>Oshikiri, T., Fukui, K., Shimodaira, H. (2016). Cross-Lingual Word Representations via Spectral Graph Embeddings. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 493–498). Berlin, Germany. Association for Computational Linguistics.


## Contents
* `src/` : Source codes (C++ & Rcpp version)
* `cpp/` : Source codes (C++ only version)
* `experiments/` : Code used in the experiments
* `tools/`


## Implemented methods
* CL-LSI [Littman+ 1998]
* Eigenwords [Dhillon+ 2012] [Dhillon+ 2015]
    * One-Step CCA (OSCCA)
    * Two-Step CCA (TSCCA)
* **Eigendocs**
* **Cross-Lingual Eigenwords** (CL-Eigenwords) [Oshikiri+ 2016]


## Required datasets for experiments
* Reuters Corpora (RCV1) <http://trec.nist.gov/data/reuters/reuters.html>
* Europarl corpus <http://www.statmt.org/europarl/>
* questions-words.txt <https://code.google.com/archive/p/word2vec/>
* The WordSimilarity-353 Test Collection <http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>
* text8 corpus <http://mattmahoney.net/dc/text8.zip>


## Submodules
* [ntessore/redsvd-h - GitHub](https://github.com/ntessore/redsvd-h)
* [docopt/docopt.cpp - GitHub](https://github.com/docopt/docopt.cpp/)
* [word2vec - Google Codes](https://code.google.com/archive/p/word2vec/)


## References
* Oshikiri, T., Fukui, K., Shimodaira, H. (2016). Cross-Lingual Word Representations via Spectral Graph Embeddings. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers) (pp. 493–498). Berlin, Germany. Association for Computational Linguistics. [[pdf](http://www.aclweb.org/anthology/P/P16/P16-2080.pdf), [bib](http://www.aclweb.org/anthology/P/P16/P16-2080.bib)]
* Dhillon, P., Rodu, J., Foster, D., and Ungar, L. (2012). Two step cca: A new spectral method for estimating vector models of words. In Langford, J. and Pineau, J., editors, Proceedings of the 29th International Conference on Machine Learning (ICML-12), ICML ’12, pages 1551–1558, New York, NY, USA. Omnipress.
* Dhillon, P. S., Foster, D. P., and Ungar, L. H. (2015). Eigenwords: Spectral word embeddings. Journal of Machine Learning Research, 16:3035–3078. [[GitHub](https://github.com/paramveerdhillon/swell/), [code&data](http://www.pdhillon.com/code.html)]
* Littman, M. L., Dumais, S. T., & Landauer, T. K. (1998). Automatic cross-language information retrieval using latent semantic indexing. Cross-Language Information Retrieval, 51–62. 


## License
GPL v3
