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
using Eigen::MappedSparseMatrix;
typedef Eigen::Map<Eigen::VectorXi> MapIM;
typedef Eigen::MappedSparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> MapMatI;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dSparseMatrix;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::Triplet<int> T;


// [[Rcpp::export]]
Rcpp::List OSCCARedSVD(MapIM& sentence, int window_size, int vocab_size, int k, bool skip_null_words) {
  std::cout << "OSCCARedSVD()" << std::endl;

  unsigned long long i, j, i_sentence;
  unsigned long long sentence_size = sentence.size();
  unsigned long long c_col_size = 2*(unsigned long long)window_size*(unsigned long long)vocab_size;
  int i_offset, offset;
  int offsets[2*window_size];

  iSparseMatrix twc(vocab_size, c_col_size);  // t(W) %*% C
  twc.reserve((unsigned long long)(0.05 * vocab_size * c_col_size));
  
  VectorXi tww_diag(vocab_size), tcc_diag(c_col_size);
  tww_diag.setZero();
  tcc_diag.setZero();
  
  std::cout << "window size   = " << window_size   << std::endl;
  std::cout << "vocab size    = " << vocab_size    << std::endl;
  std::cout << "sentence size = " << sentence_size << std::endl;
  std::cout << "c_col_size    = " << c_col_size    << std::endl;
  std::cout << std::endl;

  i_offset = 0;
  for (offset=-window_size; offset<=window_size; offset++){
    if (offset != 0) {
      offsets[i_offset] = offset;
      i_offset++;
    }
  }
  
  for (i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    if (sentence[i_sentence] >= 0) {
      i = sentence[i_sentence];
      tww_diag(i) += 1;
    }
    
    for (i_offset=0; i_offset<2*window_size; i_offset++) {
      offset = offsets[i_offset];
      
      if (i_sentence + offset >= 0 && i_sentence + offset < sentence_size && sentence[i_sentence + offset] >= 0) {
        j = sentence[i_sentence + offset] + vocab_size * i_offset;

        // Skip if sentence[i_sentence] is null words and skip_null_words is true
        if (sentence[i_sentence] >= 0 || !skip_null_words){
          tcc_diag(j) += 1;
        }
        
        if (sentence[i_sentence] >= 0) {
          twc.coeffRef(i, j) += 1;
        }
      }
    }
  }
  
  twc.makeCompressed();
  
  std::cout << "Calculate Matrix A..." << std::endl;
  VectorXd tcc_h(tcc_diag.cast <double> ().cwiseInverse().cwiseSqrt());
  VectorXd tww_h(tww_diag.cast <double> ().cwiseInverse().cwiseSqrt());
  dSparseMatrix a(tww_h.asDiagonal() * (twc.cast <double> ().eval()) * tcc_h.asDiagonal());

  // Calculate Randomized SVD
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdA(a, k);

  return Rcpp::List::create(
    Rcpp::Named("A") = Rcpp::wrap(a),
    Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
    Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
    Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
    Rcpp::Named("window.size") = Rcpp::wrap(window_size),
    Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
    Rcpp::Named("skip.null.words") = Rcpp::wrap(skip_null_words)
  );
}


// [[Rcpp::export]]
Rcpp::List RedSVDEigenwords(MapIM& sentence, int window_size, int vocab_size, int k) {
  unsigned long long i, j, i_sentence, n_non_nullwords, n_added_words;
  unsigned long long sentence_size = sentence.size();
  unsigned long long c_col_size = 2*(unsigned long long)window_size*(unsigned long long)vocab_size;
  int i_offset, offset;
  int offsets[2*window_size];
  iSparseMatrix w;
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

  iSparseMatrix c(n_non_nullwords, c_col_size);
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


  // Calculate RedSVD
  
  VectorXd cww_inverse((w.transpose() * w).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  VectorXd ccc_inverse((c.transpose() * c).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
  dSparseMatrix cwc((w.transpose() * c).eval().cast <double>());
  
  dSparseMatrix a((cww_inverse.asDiagonal() * cwc * ccc_inverse.asDiagonal()));
  
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
