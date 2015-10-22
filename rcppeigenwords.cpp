/*
  RcppEigenwords
  
  sentence の要素で-1となっているのは NULL word に対応する．
*/

#include <Rcpp.h>
#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::MatrixXi;
using Eigen::VectorXi;
using Eigen::MappedSparseMatrix;
using Rcpp::List;
typedef Eigen::Map<VectorXi> MapIM;
typedef Eigen::SparseMatrix<double> dSparseMatrix;
typedef Eigen::SparseMatrix<int> iSparseMatrix;
typedef Eigen::Triplet<int> T;


// [[Rcpp::export]]
SEXP MakeMatrices(MapIM& sentence, int window_size, int vocab_size) {
  int i, j, i_sentence;
  unsigned long long sentence_size = sentence.size();
  int i_offset, offset;
  int offsets[2*window_size];
  
  dSparseMatrix w(sentence_size, (unsigned long long)vocab_size);
  dSparseMatrix c(sentence_size, 2*(unsigned long long)window_size*(unsigned long long)vocab_size);
  std::vector<T> tripletList;
  
  std::cout << "window size = "   << window_size   << std::endl;
  std::cout << "vocab size = "    << vocab_size    << std::endl;
  std::cout << "sentence size = " << sentence_size << std::endl;


  // Make word matrix
  std::cout << "Constructing word matrix..." << std::endl;
  
  tripletList.reserve(sentence_size);

  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    if (sentence[i_sentence] >= 0) {
      i = i_sentence;
      j = sentence[i_sentence];
    
      tripletList.push_back(T(i, j, 1));
    }
  }

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

  tripletList.reserve(2*(unsigned long long)window_size*sentence_size);
  
  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    for (i_offset=0; i_offset<2*window_size; i_offset++) {
      if (sentence[i_sentence] >= 0){
        i = i_sentence - offsets[i_offset];
        j = sentence[i_sentence] + i_offset*vocab_size;
        
        if (i >= 0 && i < sentence_size && j >= 0 && j < vocab_size) {
          tripletList.push_back(T(i, j, 1));
        }
      }
    }
  }
  c.setFromTriplets(tripletList.begin(), tripletList.end());
  
  
  return Rcpp::List::create(Rcpp::Named("W") = Rcpp::wrap(w),
                            Rcpp::Named("C") = Rcpp::wrap(c));
}

// [[Rcpp::export]]
SEXP MakeSVDMatrix(MappedSparseMatrix<int> x, MappedSparseMatrix<int> y) {
  
  int size_x = x.cols();
  int size_y = y.cols();
  
  iSparseMatrix cxx(x.adjoint() * x);
  iSparseMatrix cxy(x.adjoint() * y);
  iSparseMatrix cyy(y.adjoint() * y);
//  iSparseMatrix a(cxx.inverse() * cxy * cyy.inverse());

  return Rcpp::wrap(0);
//  
//  return Rcpp::wrap(a);
}