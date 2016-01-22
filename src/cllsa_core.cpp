#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Rcpp.h>
#include <RcppEigen.h>
#include "redsvd.hpp"

typedef Eigen::Triplet<int> Triplet;

const int TRIPLET_VECTOR_SIZE = 100000000;


// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]


// Rcpp implementation of Cross-Lingual Latent Semantic Analysis
//   with Randomized SVD
// [[Rcpp::export]]
Rcpp::List CLLSA(const Rcpp::IntegerVector& corpus_id_concated,
                 const Rcpp::IntegerVector& document_id_concated,
                 const Rcpp::IntegerVector vocab_sizes,
                 const Rcpp::IntegerVector sentence_lengths,
                 const int dim_common_space)
{
  unsigned long long ii_start = 0;
  unsigned long long i_concated = 0;
  
  // Construct word-document matrix from corpora
  std::vector<Triplet> word_document_matrix_tripletlist;
  word_document_matrix_tripletlist.reserve(TRIPLET_VECTOR_SIZE);
  
  for (int i_lang = 0; i_lang < vocab_sizes.length(); i_lang++) {
    const unsigned long long vocab_size = vocab_sizes[i_lang];
    const unsigned long long sentence_length = sentence_lengths[i_lang];
    
    for (unsigned long long i_word = 0; i_word < sentence_length; i_word++) {
      const unsigned long long ii = ii_start + corpus_id_concated[i_concated];
      const unsigned long long jj = document_id_concated[i_concated];
      word_document_matrix_tripletlist.push_back(Triplet(ii, jj, 1));
      i_concated++;
    }
    ii_start += vocab_size;
  }
  
  const unsigned long long nrow = Rcpp::sum(vocab_sizes);
  const unsigned long long ncol = Rcpp::max(document_id_concated) + 1;
  std::cout << nrow << " x " << ncol << "-matrix" << std::endl;
  Eigen::SparseMatrix<double> word_document_matrix(nrow, ncol);
  word_document_matrix.setFromTriplets(word_document_matrix_tripletlist.begin(), word_document_matrix_tripletlist.end());
  word_document_matrix_tripletlist.clear();
  
  // Execute Singular Value Decomposition
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<Eigen::SparseMatrix<double> > svd(word_document_matrix, dim_common_space, 20);

  return Rcpp::List::create(Rcpp::Named("word_document_matrix")     = Rcpp::wrap(word_document_matrix.cast <double> ()),
                            Rcpp::Named("word_representations")     = Rcpp::wrap(svd.matrixU()),
                            Rcpp::Named("document_representations") = Rcpp::wrap(svd.matrixV()),
                            Rcpp::Named("singular_values")          = Rcpp::wrap(svd.singularValues()));
}