
#include <iostream>
#include <time.h> 
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "redsvd.hpp"


typedef Eigen::VectorXi VectorXi;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef Eigen::Map<Eigen::VectorXi> MapVectorXi;
typedef Eigen::SparseMatrix<int, Eigen::ColMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor, std::ptrdiff_t> dSparseMatrix;
typedef Eigen::Triplet<double> Triplet;


class EigenwordsOSCCA
{
private:
  std::vector<int> id_wordtype;
  int window_size;
  int vocab_size;
  int k;
  bool debug;

  unsigned long long c_col_size;

  VectorXi tWW_diag;
  VectorXi tCC_diag;
  dSparseMatrix tWW_h_diag;
  dSparseMatrix tCC_h_diag;
  iSparseMatrix tWC;

  MatrixXd word_vectors;
  MatrixXd context_vectors;
  VectorXd singular_values;

  void construct_matrices ();

public:
  EigenwordsOSCCA(const std::vector<int>& _id_wordtype,
                  const int _window_size,
                  const int _vocab_size,
                  const int _k,
                  const bool debug);
  void compute();
  const VectorXi get_tww_diag() const { return tWW_diag; }
  const VectorXi get_tcc_diag() const { return tCC_diag; }
  dSparseMatrix get_twc() const { return tWC.cast <double> (); }
  const MatrixXd get_word_vectors() const { return word_vectors; }
  const MatrixXd get_context_vectors() const { return context_vectors; }
  const VectorXd get_singular_values() const { return singular_values; }
};


class EigenwordsTSCCA
{
private:
  std::vector<int> id_wordtype;
  int window_size;
  int vocab_size;
  int k;
  bool debug;
  
  unsigned long long c_col_size;
  unsigned long long lr_col_size;
  
  VectorXi tWW_diag;
  dSparseMatrix tWW_h_diag;
  iSparseMatrix tWC;
  iSparseMatrix tLL;
  iSparseMatrix tLR;
  iSparseMatrix tRR;
  
  MatrixXd word_vectors;
  MatrixXd context_vectors;
  VectorXd singular_values;
  
  void construct_matrices ();
  void run_tscca();
  
public:
  EigenwordsTSCCA(const std::vector<int>& _id_wordtype,
                  const int _window_size,
                  const int _vocab_size,
                  const int _k,
                  const bool debug);
  void compute();
  const VectorXi get_tww_diag() const { return tWW_diag; }
  dSparseMatrix get_twc() const { return tWC.cast <double> (); }
  const MatrixXd get_word_vectors() const { return word_vectors; }
  const MatrixXd get_context_vectors() const { return context_vectors; }
  const VectorXd get_singular_values() const { return singular_values; }
};


class Eigendocs
{
private:
  std::vector<int> id_wordtype;
  std::vector<int> id_document;
  int window_size;
  int vocab_size;
  int k;
  bool link_w_d;
  bool link_c_d;
  bool debug;

  unsigned long long c_col_size;
  unsigned long long lr_col_size;

  VectorXi tWW_diag;
  VectorXi tCC_diag;
  VectorXi tDD_diag;
  iSparseMatrix H;
  VectorXi G_diag;

  MatrixXd vector_representations;
  VectorXd eigenvalues;

  void construct_matrices ();

public:
  unsigned long long p;
  unsigned long long p_indices[3];
  unsigned long long p_head_domains[3];

  Eigendocs(const std::vector<int>& _id_wordtype,
            const std::vector<int>& _id_document,
            const int _window_size,
            const int _vocab_size,
            const int _k,
            const bool _link_w_d,
            const bool _link_c_d,
            const bool debug);
  void compute();
  const VectorXi get_g_diag() const { return G_diag; }
  dSparseMatrix get_h() const { return H.cast <double> (); }
  const MatrixXd get_vector_representations() const { return vector_representations; }
  const VectorXd get_eigenvalues() const { return eigenvalues; }
};


class CLEigenwords
{
private:
  std::vector<int> id_wordtype_concated;
  std::vector<int> id_document_concated;
  std::vector<int> window_sizes;
  std::vector<int> vocab_sizes;
  std::vector<unsigned long long> id_wordtype_lengths;
  int k;
  bool link_v_c;
  std::vector<double> weight_vsdoc;
  bool debug;

  int n_languages;
  unsigned long long n_documents;
  unsigned long long n_domain;
  unsigned long long p;
  std::vector<unsigned long long> p_indices;
  std::vector<unsigned long long> p_head_domains;

  std::vector<unsigned long long> c_col_sizes;
  std::vector<unsigned long long> lr_col_sizes;
  std::vector<std::vector<double> >  inverse_word_count_table;

  VectorXd G_diag;
  dSparseMatrix H;

  MatrixXd vector_representations;
  VectorXd eigenvalues;
  VectorXd eigenvalues_all;

  void construct_matrices();

public:
  CLEigenwords(const std::vector<int>& _id_wordtype_concated,
               const std::vector<int>& _id_document_concated,
               const std::vector<int> _window_sizes,
               const std::vector<int> _vocab_sizes,
               const std::vector<unsigned long long> _id_wordtype_lengths,
               const int _k,
               const bool _link_v_c,
               const std::vector<double> _weight_vsdoc,
               const bool debug
  );
  void compute(int dimension_evd);
  const VectorXd get_g_diag() const { return G_diag; }
  dSparseMatrix get_h() const { return H; }
  const MatrixXd get_vector_representations() const { return vector_representations; }
  const VectorXd get_eigenvalues() const { return eigenvalues; }
  const VectorXd get_eigenvalues_all() const { return eigenvalues_all; }
  int get_n_domain() const { return n_domain; }
  unsigned long long get_p() const { return p; }
  unsigned long long get_p_head_domains(int index) const { return p_head_domains[index]; }
};


template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX);
void fill_offset_table (int offsets[], int window_size);
void construct_h_diag_matrix (VectorXi &tXX_diag, dSparseMatrix &tXX_h_diag);
void construct_h_diag_matrix_double (VectorXd &tXX_diag, dSparseMatrix &tXX_h_diag);

