/*
 * kadingir_rcpp_wrapper.cpp 
 *
 * kadingir_core.cpp の Rcpp wrapper
 */


#include <Rcpp.h>
#include <RcppEigen.h>
#include "kadingir_core.hpp"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]


// [[Rcpp::export]]
Rcpp::List EigenwordsRedSVD(const MapVectorXi& sentence,
                            const int window_size,
                            const int vocab_size,
                            const int k,
                            const bool mode_oscca)
{
  EigenwordsResults eigenwords = EigenwordsRedSVD_cpp(sentence, window_size, vocab_size, k, mode_oscca);

  return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(eigenwords.get_word_vectors()),
                            Rcpp::Named("context_vector") = Rcpp::wrap(eigenwords.get_context_vectors()),
                            Rcpp::Named("D") = Rcpp::wrap(eigenwords.get_singular_values())
                            );
}
