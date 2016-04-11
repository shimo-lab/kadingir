kadingir: Implementation of some methods of word embedding
=====================================================

## Implemented methods
* CL-LSI [Littman+ 1998]
* Eigenwords [Dhillon+ 2012] [Dhillon+ 2015]
    * One-Step CCA (OSCCA)
    * Two-Step CCA (TSCCA)
* **Eigendocs**
* **Cross-Lingual Eigenwords** (CL-Eigenwords)


## Required datasets for experiments
* Reuters Corpora (RCV1)
    * <http://trec.nist.gov/data/reuters/reuters.html>
* Europarl corpus
    * <http://www.statmt.org/wmt11/translation-task.html>
    * <http://www.statmt.org/europarl/>
* questions-words.txt
    * From <https://code.google.com/archive/p/word2vec/>
* The WordSimilarity-353 Test Collection
    * <http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>
* text8 corpus <http://mattmahoney.net/dc/text8.zip>


## References
### Papers
* Dhillon, P., Rodu, J., Foster, D., and Ungar, L. (2012). Two step cca: A new spectral method for estimating vector models of words. In Langford, J. and Pineau, J., editors, Proceedings of the 29th International Conference on Machine Learning (ICML-12), ICML ’12, pages 1551–1558, New York, NY, USA. Omnipress.
* Dhillon, P. S., Foster, D. P., and Ungar, L. H. (2015). Eigenwords: Spectral word embeddings. Journal of Machine Learning Research, 16:3035–3078.
    * Source code : [paramveerdhillon/swell - GitHub](https://github.com/paramveerdhillon/swell/)
    * Output : [Codes | Paramveer Dhillon](http://www.pdhillon.com/code.html)
* Littman, M. L., Dumais, S. T., & Landauer, T. K. (1998). Automatic cross-language information retrieval using latent semantic indexing. Cross-Language Information Retrieval, 51–62. 
    
### Submodules
* [ntessore/redsvd-h - GitHub](https://github.com/ntessore/redsvd-h)
* [docopt/docopt.cpp - GitHub](https://github.com/docopt/docopt.cpp/)
