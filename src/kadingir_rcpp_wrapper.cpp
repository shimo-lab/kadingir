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
Rcpp::List EigenwordsOSCCACpp(
    const Rcpp::IntegerVector& sentence,
    const int window_size,
    const int vocab_size,
    const int k,
    const bool debug
  )
{
  std::vector<int> sentence_stdvector = Rcpp::as<std::vector<int> >(sentence);

  EigenwordsOSCCA eigenwords = EigenwordsOSCCA(sentence_stdvector, window_size, vocab_size, k, debug);
  eigenwords.compute();

  return Rcpp::List::create(
    Rcpp::Named("tWW_diag")       = Rcpp::wrap(eigenwords.get_tww_diag()),
    Rcpp::Named("tCC_diag")       = Rcpp::wrap(eigenwords.get_tcc_diag()),
    Rcpp::Named("tWC")            = Rcpp::wrap(eigenwords.get_twc()),
    Rcpp::Named("word_vector")    = Rcpp::wrap(eigenwords.get_word_vectors()),
    Rcpp::Named("context_vector") = Rcpp::wrap(eigenwords.get_context_vectors()),
    Rcpp::Named("singular_values")= Rcpp::wrap(eigenwords.get_singular_values())
    );
}


// [[Rcpp::export]]
Rcpp::List EigenwordsTSCCACpp(
    const Rcpp::IntegerVector& sentence,
    const int window_size,
    const int vocab_size,
    const int k,
    const bool debug
)
{
  std::vector<int> sentence_stdvector = Rcpp::as<std::vector<int> >(sentence);
  
  EigenwordsTSCCA eigenwords = EigenwordsTSCCA(sentence_stdvector, window_size, vocab_size, k, debug);
  eigenwords.compute();
  
  return Rcpp::List::create(
    Rcpp::Named("tWW_diag")       = Rcpp::wrap(eigenwords.get_tww_diag()),
    Rcpp::Named("tWC")            = Rcpp::wrap(eigenwords.get_twc()),
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
    const bool link_w_d,
    const bool link_c_d,
    const bool debug
  )
{
  std::vector<int> sentence_stdvector = Rcpp::as<std::vector<int> >(sentence);
  std::vector<int> document_id_stdvector = Rcpp::as<std::vector<int> >(document_id);
  
  Eigendocs eigendocs = Eigendocs(sentence_stdvector, document_id_stdvector, window_size, vocab_size, k,
                                  link_w_d, link_c_d, debug);
  eigendocs.compute();

  Rcpp::NumericVector p_head_domains_return(3);
  for (int i = 0; i < 3; i++){
    p_head_domains_return[i] = eigendocs.p_head_domains[i];
  }


  return Rcpp::List::create(
    Rcpp::Named("G_diag") = Rcpp::wrap(eigendocs.get_g_diag()),
    Rcpp::Named("H") = Rcpp::wrap(eigendocs.get_h()),
    Rcpp::Named("V") = Rcpp::wrap(eigendocs.get_vector_representations()),
    Rcpp::Named("eigenvalues") = Rcpp::wrap(eigendocs.get_eigenvalues()),
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
    const bool link_w_d,
    const bool link_c_d,
    const bool weighting_tf,
    const Rcpp::NumericVector weight_vsdoc,
    const bool debug
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
                                           link_w_d, link_c_d,
                                           weighting_tf, weight_vsdoc_stdvector, debug);
  cleigenwords.compute();

  int n_domain = cleigenwords.get_n_domain();
  Rcpp::NumericVector p_head_domains_return(n_domain);
  for (int i = 0; i < n_domain; i++){
    p_head_domains_return[i] = cleigenwords.get_p_head_domains(i);
  }

  return Rcpp::List::create(
    Rcpp::Named("G_diag") = Rcpp::wrap(cleigenwords.get_g_diag()),
    Rcpp::Named("H") = Rcpp::wrap(cleigenwords.get_h()),
    Rcpp::Named("V") = Rcpp::wrap(cleigenwords.get_vector_representations()),
    Rcpp::Named("eigenvalues") = Rcpp::wrap(cleigenwords.get_eigenvalues()),
    Rcpp::Named("p_head_domains") = p_head_domains_return,
    Rcpp::Named("p") = (double)cleigenwords.get_p()
    );
}
