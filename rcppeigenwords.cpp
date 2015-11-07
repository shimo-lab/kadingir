/*
  RcppEigenwords
  
  sentence の要素で-1となっているのは NULL word に対応する．
*/

#include <Rcpp.h>
#include <RcppEigen.h>
#include "redsvd.hpp"

// [[Rcpp::depends(RcppEigen)]]

using Eigen::MatrixXd;
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
Rcpp::List EigenwordsRedSVD(MapIM& sentence, int window_size, int vocab_size, int k, bool skip_null_words, bool mode_oscca) {
  
  unsigned long long i, j, i_sentence, j2;
  unsigned long long sentence_size = sentence.size();
  unsigned long long c_col_size = 2*(unsigned long long)window_size*(unsigned long long)vocab_size;
  long long i_word1, i_word2;
  int i_offset, offset;
  int offsets[2*window_size];
  bool non_null_word;

  iSparseMatrix twc(vocab_size, c_col_size);  // t(W) %*% C
  iSparseMatrix tcc(c_col_size, c_col_size);  // t(C) %*% C
  twc.reserve((unsigned long long)(0.05 * vocab_size * c_col_size));
  tcc.reserve((unsigned long long)(0.05 * c_col_size * c_col_size));
  MatrixXd phi_l, phi_r;
  
  VectorXi tww_diag(vocab_size);
  VectorXi tcc_diag(c_col_size);
  tww_diag.setZero();  
  tcc_diag.setZero();
  
  std::cout << "mode          = ";
  if (mode_oscca) {
    std::cout << "OSCCA" << std::endl;
  } else {
    std::cout << "TSCCA" << std::endl;
  }
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
      i_word1 = i_sentence + offsets[i_offset];
      if (i_word1 >= 0 && i_word1 < sentence_size && sentence[i_word1] >= 0) {
        j = sentence[i_word1] + vocab_size * i_offset;

        // Skip if sentence[i_sentence] is null words and skip_null_words is true
        if (sentence[i_sentence] >= 0 || !skip_null_words){
          if (mode_oscca) {
            tcc_diag(j) += 1;
          } else {
            for (int i_offset2 = 0; i_offset2<2*window_size; i_offset2++) {
              i_word2 = i_sentence + offsets[i_offset2];
              if (i_word2 >= 0 && i_word2 < sentence_size && sentence[i_word2] >= 0) {
                j2 = sentence[i_word2] + vocab_size * i_offset2;
                tcc.coeffRef(j, j2) += 1;
              }
            }
          }
        }
        
        if (sentence[i_sentence] >= 0) {
          twc.coeffRef(i, j) += 1;
        }
      }
    }
  }
  
  twc.makeCompressed();
  
  std::cout << "Calculate OSCCA/TSCCA..." << std::endl;
    
  if (mode_oscca) {
    VectorXd tww_h(tww_diag.cast <double> ().cwiseInverse().cwiseSqrt());
    VectorXd tcc_h(tcc_diag.cast <double> ().cwiseInverse().cwiseSqrt());
    dSparseMatrix a(tww_h.asDiagonal() * (twc.cast <double> ().eval()) * tcc_h.asDiagonal());
  
    // Calculate Randomized SVD
    std::cout << "Calculate Randomized SVD..." << std::endl;
    RedSVD::RedSVD<dSparseMatrix> svdA(a, k);
    
    return Rcpp::List::create(
      Rcpp::Named("tWC") = Rcpp::wrap(twc.cast <double> ()),
      Rcpp::Named("tWW_h") = Rcpp::wrap(tww_h),
      Rcpp::Named("tCC_h") = Rcpp::wrap(tcc_h),
      Rcpp::Named("A") = Rcpp::wrap(a),
      Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
      Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
      Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
      Rcpp::Named("skip.null.words") = Rcpp::wrap(skip_null_words),
      Rcpp::Named("k") = Rcpp::wrap(k)
    );
    
  } else {
    tcc.makeCompressed();
    
    // TSCCA : Step 1
    VectorXd tll_h(tcc.topLeftCorner(c_col_size/2, c_col_size/2).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
    VectorXd trr_h(tcc.bottomRightCorner(c_col_size/2, c_col_size/2).eval().diagonal().cast <double> ().cwiseInverse().cwiseSqrt());
    dSparseMatrix b(tll_h.asDiagonal() * (tcc.topRightCorner(c_col_size/2, c_col_size/2).cast <double> ().eval()) * trr_h.asDiagonal());
    
    std::cout << "Calculate Randomized SVD (1/2)..." << std::endl;
    RedSVD::RedSVD<dSparseMatrix> svdB(b, k);
    
    // TSCCA : Step 2
    phi_l = svdB.matrixU();
    phi_r = svdB.matrixV();
        
    VectorXd tww_h(tww_diag.cast <double> ().cwiseInverse().cwiseSqrt());
    VectorXd tss_h1((phi_l.transpose() * tcc.topLeftCorner(c_col_size/2, c_col_size/2).eval().cast <double> () * phi_l).eval().diagonal().cwiseInverse().cwiseSqrt());
    VectorXd tss_h2((phi_r.transpose() * tcc.bottomRightCorner(c_col_size/2, c_col_size/2).eval().cast <double> () * phi_r).eval().diagonal().cwiseInverse().cwiseSqrt());
    VectorXd tss_h(2*k);
    tss_h << tss_h1, tss_h2;
    MatrixXd tws(vocab_size, 2*k);
    tws << twc.topLeftCorner(vocab_size, c_col_size/2).cast <double> () * phi_l, twc.topRightCorner(vocab_size, c_col_size/2).cast <double> () * phi_r;
    MatrixXd a(tww_h.asDiagonal() * tws * tss_h.asDiagonal());

    std::cout << "Calculate Randomized SVD (2/2)..." << std::endl;
    RedSVD::RedSVD<MatrixXd> svdA(a, k);
    
    return Rcpp::List::create(
      Rcpp::Named("tWS") = Rcpp::wrap(tws),
      Rcpp::Named("tWW_h") = Rcpp::wrap(tww_h),
      Rcpp::Named("tSS_h") = Rcpp::wrap(tss_h),
      Rcpp::Named("A") = Rcpp::wrap(a),
      Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
      Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
      Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
      Rcpp::Named("skip.null.words") = Rcpp::wrap(skip_null_words),
      Rcpp::Named("k") = Rcpp::wrap(k)
    );
  }
}
