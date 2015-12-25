
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


class EigenwordsResults
{
private:
  MatrixXreal word_vectors;
  MatrixXreal context_vectors;
  VectorXreal singular_values;

public:
  EigenwordsResults(MatrixXreal _word_vectors, MatrixXreal _context_vectors, VectorXreal _singular_values);
  MatrixXreal get_word_vectors() { return word_vectors; }
  MatrixXreal get_context_vectors() { return context_vectors; }
  VectorXreal get_singular_values() { return singular_values; }
};


template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX);
void fill_offset_table (int offsets[], int window_size);
void construct_crossprod_matrices (const MapVectorXi& sentence,
                                   Eigen::VectorXi &tWW_diag,
                                   Eigen::VectorXi &tCC_diag,
                                   iSparseMatrix &tWC,
                                   iSparseMatrix &tLL,
                                   iSparseMatrix &tLR,
                                   iSparseMatrix &tRR,
                                   const int window_size,
                                   const int vocab_size,
                                   const bool mode_oscca);
void construct_h_diag_matrix (Eigen::VectorXi &tXX_diag, realSparseMatrix &tXX_h_diag);
EigenwordsResults EigenwordsRedSVD_cpp(const MapVectorXi& sentence,
                                       const int window_size,
                                       const int vocab_size,
                                       const int k,
                                       const bool mode_oscca);

