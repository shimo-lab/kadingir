/*
  RcppEigenwords
  
  sentence の要素で-1となっているのは NULL word に対応する．
*/

#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::MappedSparseMatrix;
using Rcpp::List;
typedef Eigen::Map<Eigen::VectorXi> MapIM;
typedef Eigen::SparseMatrix<double> dSparseMatrix;
typedef Eigen::SparseMatrix<int> iSparseMatrix;
typedef Eigen::DiagonalMatrix<int, Eigen::Dynamic> iDiagonalMatrix;
typedef Eigen::Triplet<int> T;


// [[Rcpp::export]]
Rcpp::List MakeMatrices(MapIM& sentence, int window_size, int vocab_size) {
  int i, j, i_sentence;
  unsigned long long sentence_size = sentence.size();
  unsigned long long ii, n_nonzeros;
  unsigned long long *i_nonzeros = (unsigned long long *)malloc(sentence_size * sizeof(unsigned long long));
  unsigned long long c_col_size = (unsigned long long)(2*window_size*vocab_size);
  
  int i_offset, offset;
  int offsets[2*window_size];
  
  dSparseMatrix w, c;
  std::vector<T> tripletList;
  
  std::cout << "window size = "   << window_size   << std::endl;
  std::cout << "vocab size = "    << vocab_size    << std::endl;
  std::cout << "sentence size = " << sentence_size << std::endl;


  // Make word matrix
  std::cout << "Constructing word matrix..." << std::endl;
  
  tripletList.reserve(sentence_size);

  ii = 0;
  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    if (sentence[i_sentence] >= 0) {
      i = ii;
      j = sentence[i_sentence];
      
      tripletList.push_back(T(i, j, 1));
      
      i_nonzeros[ii] = i_sentence;
      ii++;
    }
  }
  n_nonzeros = ii;

  w.resize(n_nonzeros, (unsigned long long)vocab_size);
  w.setFromTriplets(tripletList.begin(), tripletList.end());
  tripletList.clear();

  
  // Make context matrix
  std::cout << "Constructing context matrix..." << std::endl;

  i_offset = 0;
  for (offset=-window_size; offset<=window_size; offset++){
    if (offset != 0) {
      offsets[i_offset] = offset;
      i_offset++;
    }
  }

  tripletList.reserve(2*(unsigned long long)window_size*n_nonzeros);

  for (ii=0; ii<n_nonzeros; ii++) {
    i_sentence = i_nonzeros[ii];
    
    for (i_offset=0; i_offset<2*window_size; i_offset++) {
      if (sentence[i_sentence] >= 0) {
        i = i_sentence - offsets[i_offset];
        j = sentence[i_sentence] + i_offset*vocab_size;
        
        if ((i >= 0) && (i < n_nonzeros) && (j >= 0) && (j < c_col_size)) {
          tripletList.push_back(T(i, j, 1));
        }
      }
    }
  }
  c.resize(n_nonzeros, c_col_size);
  c.setFromTriplets(tripletList.begin(), tripletList.end());

  free(i_nonzeros);
  
  return Rcpp::List::create(Rcpp::Named("W") = Rcpp::wrap(w),
                            Rcpp::Named("C") = Rcpp::wrap(c));
}


// [[Rcpp::export]]
dSparseMatrix MakeSVDMatrix(MappedSparseMatrix<int> x, MappedSparseMatrix<int> y) {
  VectorXd cxx_inverse((x.transpose() * x).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  VectorXd cyy_inverse((y.transpose() * y).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  dSparseMatrix cxy((x.transpose() * y).eval().cast <double>());
  
  return cxx_inverse.asDiagonal() * cxy * cyy_inverse.asDiagonal();
}
