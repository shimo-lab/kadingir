
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "redsvd.hpp"


typedef Eigen::VectorXi VectorXi;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::Map<Eigen::VectorXi> MapVectorXi;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dSparseMatrix;
typedef Eigen::Triplet<double> Triplet;


class Eigenwords
{
private:
  std::vector<int> sentence;
  int window_size;
  int vocab_size;
  int k;
  bool mode_oscca;

  unsigned long long c_col_size;
  unsigned long long lr_col_size;

  MatrixXd word_vectors;
  MatrixXd context_vectors;
  VectorXd singular_values;

  void construct_matrices (VectorXi &tWW_diag,
                           VectorXi &tCC_diag,
                           iSparseMatrix &tWC,
                           iSparseMatrix &tLL,
                           iSparseMatrix &tLR,
                           iSparseMatrix &tRR);
  void run_oscca(dSparseMatrix &tWW_h_diag,
                 iSparseMatrix &tWC,
                 VectorXi &tCC_diag);
  void run_tscca(dSparseMatrix &tWW_h_diag,
                 iSparseMatrix &tLL,
                 iSparseMatrix &tLR,
                 iSparseMatrix &tRR,
                 iSparseMatrix &tWC);

public:
  Eigenwords(const std::vector<int>& _sentence,
             const int _window_size,
             const int _vocab_size,
             const int _k,
             const bool _mode_oscca);
  void compute();
  MatrixXd get_word_vectors() { return word_vectors; }
  MatrixXd get_context_vectors() { return context_vectors; }
  VectorXd get_singular_values() { return singular_values; }
};


class Eigendocs
{
private:
  std::vector<int> sentence;
  std::vector<int> document_id;
  int window_size;
  int vocab_size;
  int k;
  bool link_w_d;
  bool link_c_d;
  double gamma_G;
  double gamma_H;

  unsigned long long c_col_size;
  unsigned long long lr_col_size;

  MatrixXd vector_representations;
  VectorXd singular_values;

  void construct_matrices (VectorXi &tWW_diag,
                           VectorXi &tCC_diag,
                           VectorXi &tDD_diag,
                           iSparseMatrix &H);

public:
  unsigned long long p;
  unsigned long long p_indices[3];
  unsigned long long p_head_domains[3];

  Eigendocs(const std::vector<int>& _sentence,
            const std::vector<int>& _document_id,
            const int _window_size,
            const int _vocab_size,
            const int _k,
            const bool _link_w_d,
            const bool _link_c_d,
            const double _gamma_G,
            const double _gamma_H);
  void compute();
  MatrixXd get_vector_representations() { return vector_representations; }
  VectorXd get_singular_values() { return singular_values; }
};


class CLEigenwords
{
private:
  std::vector<int> sentence_concated;
  std::vector<int> document_id_concated;
  std::vector<int> window_sizes;
  std::vector<int> vocab_sizes;
  std::vector<unsigned long long> sentence_lengths;
  int k;
  double gamma_G;
  double gamma_H;
  bool link_w_d;
  bool link_c_d;
  bool weighting_tf;
  std::vector<double> weight_vsdoc;

  int n_languages;
  unsigned long long n_documents;
  unsigned long long n_domain;
  unsigned long long p;
  std::vector<unsigned long long> p_indices;
  std::vector<unsigned long long> p_head_domains;

  std::vector<unsigned long long> c_col_sizes;
  std::vector<unsigned long long> lr_col_sizes;
  std::vector<std::vector<double> >  inverse_word_count_table;

  MatrixXd vector_representations;
  VectorXd singular_values;

  void construct_inverse_word_count_table();
  void construct_matrices (VectorXd &G_diag, dSparseMatrix &H);

public:
  CLEigenwords(const std::vector<int>& _sentence_concated,
               const std::vector<int>& _document_id_concated,
               const std::vector<int> _window_sizes,
               const std::vector<int> _vocab_sizes,
               const std::vector<unsigned long long> _sentence_lengths,
               const int _k,
               const double _gamma_G,
               const double _gamma_H,
               const bool _link_w_d,
               const bool _link_c_d,
               const bool _weighting_tf,
               const std::vector<double> _weight_vsdoc
  );
  void compute();
  MatrixXd get_vector_representations() { return vector_representations; }
  VectorXd get_singular_values() { return singular_values; }
  int get_n_domain() { return n_domain; }
  unsigned long long get_p() { return p; }
  unsigned long long get_p_head_domains(int index) { return p_head_domains[index]; }
};


template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX);
void fill_offset_table (int offsets[], int window_size);
void construct_h_diag_matrix (VectorXi &tXX_diag, dSparseMatrix &tXX_h_diag);
void construct_h_diag_matrix_double (VectorXd &tXX_diag, dSparseMatrix &tXX_h_diag);

