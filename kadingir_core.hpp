
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


template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX);
void fill_offset_table (int offsets[], int window_size);
void construct_h_diag_matrix (Eigen::VectorXi &tXX_diag, realSparseMatrix &tXX_h_diag);

