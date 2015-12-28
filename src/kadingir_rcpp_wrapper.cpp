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


// [[Rcpp::export]]
Rcpp::List MCEigendocsRedSVD(const MapVectorXi& sentence_concated,
                             const MapVectorXi& document_id_concated,
                             const Rcpp::IntegerVector window_sizes,
                             const Rcpp::IntegerVector vocab_sizes,
                             const Rcpp::IntegerVector sentence_lengths,
                             const int k,
                             const real gamma_G,
                             const real gamma_H,
                             const bool link_w_d,
                             const bool link_c_d,
                             const bool doc_weighting,
                             const real weight_doc_vs_vc
                             )
{
  Eigen::VectorXi window_sizes_eigen(window_sizes.length());
  for (int i = 0; i < window_sizes.length(); i++) {
    window_sizes_eigen(i) = window_sizes[i];
  }

  Eigen::VectorXi vocab_sizes_eigen(vocab_sizes.length());
  for (int i = 0; i < vocab_sizes.length(); i++) {
    vocab_sizes_eigen(i) = vocab_sizes[i];
  }

  Eigen::VectorXi sentence_lengths_eigen(sentence_lengths.length());
  for (int i = 0; i < sentence_lengths.length(); i++) {
    sentence_lengths_eigen(i) = sentence_lengths[i];
  }


  MCEigendocs mceigendocs = MCEigendocs(sentence_concated, document_id_concated,
                                        window_sizes_eigen, vocab_sizes_eigen,
                                        sentence_lengths_eigen, k,
                                        link_w_d, link_c_d, gamma_G, gamma_H,
                                        doc_weighting, weight_doc_vs_vc);
  mceigendocs.compute();

  int n_domain = mceigendocs.get_n_domain();
  Rcpp::NumericVector p_head_domains_return(n_domain);
  for (int i = 0; i < n_domain; i++){
    p_head_domains_return[i] = mceigendocs.get_p_head_domains(i);
  }

  return Rcpp::List::create(Rcpp::Named("V") = Rcpp::wrap(mceigendocs.get_vector_representations()),
                            Rcpp::Named("singular_values") = Rcpp::wrap(mceigendocs.get_singular_values()),
                            Rcpp::Named("p_head_domains") = p_head_domains_return,
                            Rcpp::Named("p") = (double)mceigendocs.get_p()
                            );
}
