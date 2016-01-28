/*
 * kadingir_rcpp_wrapper.cpp :
 *
 *     Rcpp wrapper of kadingir_core.cpp
 */


#include <iostream>
#include <Rcpp.h>
#include <RcppEigen.h>
#include "kadingir_core.hpp"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]


// [[Rcpp::export]]
Rcpp::List EigenwordsCpp(
    const Rcpp::IntegerVector& sentence,
    const int window_size,
    const int vocab_size,
    const int k,
    const bool mode_oscca,
    const bool debug
  )
{
  std::vector<int> sentence_stdvector = Rcpp::as<std::vector<int> >(sentence);
  
  Eigenwords eigenwords = Eigenwords(sentence_stdvector, window_size, vocab_size, k, mode_oscca, debug);
  eigenwords.compute();

  return Rcpp::List::create(
    Rcpp::Named("word_vector")    = Rcpp::wrap(eigenwords.get_word_vectors()),
    Rcpp::Named("context_vector") = Rcpp::wrap(eigenwords.get_context_vectors()),
    Rcpp::Named("D")              = Rcpp::wrap(eigenwords.get_singular_values())
    );
}


// [[Rcpp::export]]
Rcpp::List EigendocsCpp(
    const Rcpp::IntegerVector& sentence,
    const Rcpp::IntegerVector& document_id,
    const int window_size,
    const int vocab_size,
    const int k,
    const double gamma_G,
    const double gamma_H,
    const bool link_w_d,
    const bool link_c_d
  )
{
  std::vector<int> sentence_stdvector = Rcpp::as<std::vector<int> >(sentence);
  std::vector<int> document_id_stdvector = Rcpp::as<std::vector<int> >(document_id);
  
  Eigendocs eigendocs = Eigendocs(sentence_stdvector, document_id_stdvector, window_size, vocab_size, k,
                                  link_w_d, link_c_d, gamma_G, gamma_H);
  eigendocs.compute();

  Rcpp::NumericVector p_head_domains_return(3);
  for (int i = 0; i < 3; i++){
    p_head_domains_return[i] = eigendocs.p_head_domains[i];
  }


  return Rcpp::List::create(
    Rcpp::Named("V") = Rcpp::wrap(eigendocs.get_vector_representations()),
    Rcpp::Named("singular_values") = Rcpp::wrap(eigendocs.get_singular_values()),
    Rcpp::Named("p_head_domains") = p_head_domains_return,
    Rcpp::Named("p") = (double)eigendocs.p
    );
}


// [[Rcpp::export]]
Rcpp::List CLEigenwordsCpp(
    const Rcpp::IntegerVector& sentence_concated,
    const Rcpp::IntegerVector& document_id_concated,
    const Rcpp::IntegerVector window_sizes,
    const Rcpp::IntegerVector vocab_sizes,
    const Rcpp::IntegerVector sentence_lengths,
    const int k,
    const double gamma_G,
    const double gamma_H,
    const bool link_w_d,
    const bool link_c_d,
    const bool weighting_tf,
    const Rcpp::NumericVector weight_vsdoc
  )
{
  std::vector<int> sentence_concated_stdvector = Rcpp::as<std::vector<int> >(sentence_concated);
  std::vector<int> document_id_concated_stdvector = Rcpp::as<std::vector<int> >(document_id_concated);
  std::vector<int> window_sizes_stdvector = Rcpp::as<std::vector<int> >(window_sizes);
  std::vector<int> vocab_sizes_stdvector = Rcpp::as<std::vector<int> >(vocab_sizes);
  std::vector<double> weight_vsdoc_stdvector = Rcpp::as<std::vector<double> >(weight_vsdoc);

  // Rcpp::as cannot convert Rcpp::IntegerVector to std::vector<unsigned long long>
  std::vector<unsigned long long> sentence_lengths_stdvector(sentence_lengths.length());
  for (int i = 0; i < sentence_lengths.length(); i++) {
    sentence_lengths_stdvector[i] = (unsigned long long)sentence_lengths[i];
  }
  
  CLEigenwords cleigenwords = CLEigenwords(sentence_concated_stdvector, document_id_concated_stdvector,
                                           window_sizes_stdvector, vocab_sizes_stdvector,
                                           sentence_lengths_stdvector, k,
                                           link_w_d, link_c_d, gamma_G, gamma_H,
                                           weighting_tf, weight_vsdoc_stdvector);
  cleigenwords.compute();

  int n_domain = cleigenwords.get_n_domain();
  Rcpp::NumericVector p_head_domains_return(n_domain);
  for (int i = 0; i < n_domain; i++){
    p_head_domains_return[i] = cleigenwords.get_p_head_domains(i);
  }

  return Rcpp::List::create(
    Rcpp::Named("V") = Rcpp::wrap(cleigenwords.get_vector_representations()),
    Rcpp::Named("singular_values") = Rcpp::wrap(cleigenwords.get_singular_values()),
    Rcpp::Named("p_head_domains") = p_head_domains_return,
    Rcpp::Named("p") = (double)cleigenwords.get_p()
    );
}
