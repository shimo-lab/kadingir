/*
  RcppEigenwords
  
  sentence の要素で-1となっているのは NULL word に対応する．
*/

#include <Rcpp.h>
#include <RcppEigen.h>
#include "redsvd.hpp"

// [[Rcpp::depends(RcppEigen)]]

using Eigen::MatrixXi;
using Eigen::VectorXd;
using Eigen::VectorXi;
typedef Eigen::Map<Eigen::VectorXi> MapIM;
typedef Eigen::MappedSparseMatrix<int, Eigen::RowMajor> MapMatI;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dSparseMatrix;
typedef Eigen::Triplet<int> T;


// [[Rcpp::export]]
Rcpp::List MakeMatrices(MapIM& sentence, int window_size, int vocab_size) {
  unsigned long long i, j, i_sentence, n_non_nullwords, n_added_words;
  unsigned long long sentence_size = sentence.size();
  unsigned long long c_col_size = 2*(unsigned long long)window_size*(unsigned long long)vocab_size;
  int i_offset, offset;
  int offsets[2*window_size];
  dSparseMatrix w;
  std::vector<T> tripletList;
  
  std::cout << "window size   = " << window_size   << std::endl;
  std::cout << "vocab size    = " << vocab_size    << std::endl;
  std::cout << "sentence size = " << sentence_size << std::endl;
  std::cout << "c_col_size    = " << c_col_size    << std::endl;
  std::cout << std::endl;


  // Make word matrix
  std::cout << "Constructing word matrix..." << std::endl << std::endl;
  
  tripletList.reserve(sentence_size);

  n_non_nullwords = 0;
  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    if (sentence[i_sentence] >= 0) {
      i = n_non_nullwords;
      j = sentence[i_sentence];
      
      tripletList.push_back(T(i, j, 1));

      n_non_nullwords++;
    }
  }

  w.resize(n_non_nullwords, (unsigned long long)vocab_size);
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

  dSparseMatrix c(n_non_nullwords, 2*(unsigned long long)window_size*(unsigned long long)vocab_size);
  std::cout << "before VectorXi" << std::endl;  
  VectorXi cc(VectorXi::Constant(n_non_nullwords, 2*window_size));
  std::cout << "before c.reserve()" << std::endl;
  c.reserve(cc);
  std::cout << "after  c.reserve()" << std::endl;

  n_added_words = 0;
  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    if (sentence[i_sentence] >= 0) {  // If sentence[i_sentence] is NOT null words
      for (i_offset=0; i_offset<2*window_size; i_offset++) {
        // If `i_sentence + offsets[i_offset]` is valid index of sentence
        //    and sentence[i_sentence + offsets[i_offset]] is non-null word
        if ((i_sentence + offsets[i_offset] >= 0) &&
            (i_sentence + offsets[i_offset] < sentence_size) && 
            sentence[i_sentence + offsets[i_offset]] > -1) {
          
          i = n_added_words;
          j = sentence[i_sentence + offsets[i_offset]] + i_offset*vocab_size;
            
          if ((i < n_non_nullwords) && (j < c_col_size)) {
            c.insert(i, j) = 1;
          }
        }
      }
      n_added_words++;
    }
  }

  std::cout << "before c.makeCompressed()" << std::endl;
  c.makeCompressed();
  std::cout << "after c.makeCompressed()" << std::endl;


  return Rcpp::List::create(Rcpp::Named("W") = Rcpp::wrap(w),
                            Rcpp::Named("C") = Rcpp::wrap(c));
}


// [[Rcpp::export]]
dSparseMatrix MakeSVDMatrix(MapMatI x, MapMatI y) {
  VectorXd cxx_inverse((x.transpose() * x).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  VectorXd cyy_inverse((y.transpose() * y).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  dSparseMatrix cxy((x.transpose() * y).eval().cast <double>());
  
  return (cxx_inverse.asDiagonal() * cxy * cyy_inverse.asDiagonal());
}


// [[Rcpp::export]]
Rcpp::List RedsvdOSCCA(MapMatI x, MapMatI y, int k) {
  std::cout << "a(MakeSVDMatrix(x, y))" << std::endl;
  dSparseMatrix a(MakeSVDMatrix(x, y));
  
  std::cout << "Calculate RedSVD" << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdA(a, k);
  std::cout << "after RedSVD" << std::endl;


  return Rcpp::List::create(Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
			    Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
			    Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
			    Rcpp::Named("k") = Rcpp::wrap(k),
			    Rcpp::Named("A") = Rcpp::wrap(a)
			    );
}
