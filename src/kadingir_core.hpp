
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "redsvd.hpp"


typedef double real;
typedef Eigen::VectorXd VectorXreal;
typedef Eigen::MatrixXd MatrixXreal;
typedef Eigen::Map<Eigen::VectorXi> MapVectorXi;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::SparseMatrix<real, Eigen::RowMajor, std::ptrdiff_t> realSparseMatrix;
typedef Eigen::Triplet<real> Triplet;


class Eigenwords
{
private:
  MapVectorXi sentence;
  int window_size;
  int vocab_size;
  int k;
  bool mode_oscca;

  unsigned long long c_col_size;
  unsigned long long lr_col_size;

  MatrixXreal word_vectors;
  MatrixXreal context_vectors;
  VectorXreal singular_values;

  void construct_matrices (Eigen::VectorXi &tWW_diag,
                           Eigen::VectorXi &tCC_diag,
                           iSparseMatrix &tWC,
                           iSparseMatrix &tLL,
                           iSparseMatrix &tLR,
                           iSparseMatrix &tRR);
  void run_oscca(realSparseMatrix &tWW_h_diag,
                 iSparseMatrix &tWC,
                 Eigen::VectorXi &tCC_diag);
  void run_tscca(realSparseMatrix &tWW_h_diag,
                 iSparseMatrix &tLL,
                 iSparseMatrix &tLR,
                 iSparseMatrix &tRR,
                 iSparseMatrix &tWC);

public:
  Eigenwords(const MapVectorXi& _sentence,
             const int _window_size,
             const int _vocab_size,
             const int _k,
             const bool _mode_oscca);
  void compute();
  MatrixXreal get_word_vectors();
  MatrixXreal get_context_vectors();
  VectorXreal get_singular_values();
};


class Eigendocs
{
private:
  MapVectorXi sentence;
  MapVectorXi document_id;
  int window_size;
  int vocab_size;
  int k;
  bool link_w_d;
  bool link_c_d;
  real gamma_G;
  real gamma_H;

  unsigned long long c_col_size;
  unsigned long long lr_col_size;

  MatrixXreal vector_representations;
  VectorXreal singular_values;

  void construct_matrices (Eigen::VectorXi &tWW_diag,
                           Eigen::VectorXi &tCC_diag,
                           Eigen::VectorXi &tDD_diag,
                           iSparseMatrix &H);

public:
  unsigned long long p;
  unsigned long long p_indices[3];
  unsigned long long p_head_domains[3];

  Eigendocs(const MapVectorXi& _sentence,
            const MapVectorXi& _document_id,
            const int _window_size,
            const int _vocab_size,
            const int _k,
            const bool _link_w_d,
            const bool _link_c_d,
            const real _gamma_G,
            const real _gamma_H);
  void compute();
  MatrixXreal get_vector_representations();
  VectorXreal get_singular_values();
};


class MCEigendocs
{
private:
  MapVectorXi sentence_concated;
  MapVectorXi document_id_concated;
  Eigen::VectorXi window_sizes;
  Eigen::VectorXi vocab_sizes;
  Eigen::VectorXi sentence_lengths;
  int k;
  real gamma_G;
  real gamma_H;
  bool link_w_d;
  bool link_c_d;
  bool doc_weighting;
  real weight_doc_vs_vc;

  int n_languages;
  unsigned long long n_documents;
  unsigned long long n_domain;
  unsigned long long p;
  std::vector<unsigned long long> p_indices;
  std::vector<unsigned long long> p_head_domains;

  std::vector<unsigned long long> c_col_sizes;
  std::vector<unsigned long long> lr_col_sizes;
  std::vector<real>  inverse_word_count_table;

  MatrixXreal vector_representations;
  VectorXreal singular_values;

  void construct_matrices (VectorXreal &G_diag, realSparseMatrix &H);

public:
  MCEigendocs(const MapVectorXi& _sentence_concated,
              const MapVectorXi& _document_id_concated,
              const Eigen::VectorXi _window_sizes,
              const Eigen::VectorXi _vocab_sizes,
              const Eigen::VectorXi _sentence_lengths,
              const int _k,
              const real _gamma_G,
              const real _gamma_H,
              const bool _link_w_d,
              const bool _link_c_d,
              const bool _doc_weighting,
              const real _weight_doc_vs_vc
              );
  void compute();
  MatrixXreal get_vector_representations();
  VectorXreal get_singular_values();
  int get_n_domain();
  unsigned long long get_p();
  unsigned long long get_p_head_domains(int index);
};


template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX);
void fill_offset_table (int offsets[], int window_size);
void construct_h_diag_matrix (Eigen::VectorXi &tXX_diag, realSparseMatrix &tXX_h_diag);

