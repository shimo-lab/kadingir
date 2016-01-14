kadingir : Implementation of Eigenwords and its extensions
================

# Contents
* Eigenwords [Dhillon+ 2012] [Dhillon+ 2015]
    * One-Step CCA (OSCCA)
    * Two-Step CCA (TSCCA)
* Eigendocs
* **Cross-Lingual Eigenwords** (CL-Eigenwords)

# Required datasets for experiments
* Reuters Corpora (RCV1)
    * <http://trec.nist.gov/data/reuters/reuters.html>
* Europarl corpus
    * <http://www.statmt.org/wmt11/translation-task.html>
    * <http://www.statmt.org/europarl/>

# References
## Papers
* Dhillon, Paramveer S, Jordan Rodu, Dean P Foster, and Lyle H Ungar. 2012. “Two Step CCA: A New Spectral Method for Estimating Vector Models of Words.” Proceedings of the 29th International Conference on Machine Learning (ICML-12), 1551–58.
* Dhillon, Paramveer S, Dean P Foster, and Lyle H Ungar. 2015. “Eigenwords: Spectral Word Embeddings.” Journal of Machine Learning Research 16: 1–36. 
    * Source code : [paramveerdhillon/swell - GitHub](https://github.com/paramveerdhillon/swell/)
    * Output : [Codes | Paramveer Dhillon](http://www.pdhillon.com/code.html)
    
## Submodules
* [ntessore/redsvd-h - GitHub](https://github.com/ntessore/redsvd-h)

## Test data
* test/questions-words.txt : <https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt>
* test/conbined.csv : <http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/>

