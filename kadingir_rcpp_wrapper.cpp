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
  Eigenwords eigenwords = Eigenwords(sentence, window_size, vocab_size, k, mode_oscca);
  eigenwords.compute();

  return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(eigenwords.get_word_vectors()),
                            Rcpp::Named("context_vector") = Rcpp::wrap(eigenwords.get_context_vectors()),
                            Rcpp::Named("D") = Rcpp::wrap(eigenwords.get_singular_values())
                            );
}


// [[Rcpp::export]]
Rcpp::List EigendocsRedSVD(const MapVectorXi& sentence,
                           const MapVectorXi& document_id,
                           const int window_size,
                           const int vocab_size,
                           const int k,
                           const real gamma_G,
                           const real gamma_H,
                           const bool link_w_d,
                           const bool link_c_d
                           )
{
  Eigendocs eigendocs = Eigendocs(sentence, document_id, window_size, vocab_size, k,
                                  link_w_d, link_c_d, gamma_G, gamma_H);
  eigendocs.compute();

  Rcpp::NumericVector p_head_domains_return(3);
  for (int i = 0; i < 3; i++){
    p_head_domains_return[i] = eigendocs.p_head_domains[i];
  }


  return Rcpp::List::create(Rcpp::Named("V") = Rcpp::wrap(eigendocs.get_vector_representations()),
                            Rcpp::Named("singular_values") = Rcpp::wrap(eigendocs.get_singular_values()),
                            Rcpp::Named("p_head_domains") = p_head_domains_return,
                            Rcpp::Named("p") = (double)eigendocs.p
                            );
}
