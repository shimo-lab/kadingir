/*
RcppEigenwords

sentence の要素で 0 となっているのは <OOV> (Out of Vocabulary) に対応する．
*/

#include <Rcpp.h>
#include <RcppEigen.h>
#include "redsvd.hpp"

// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::plugins(openmp)]]

typedef double real;
typedef Eigen::VectorXd VectorXreal;
typedef Eigen::MatrixXd MatrixXreal;
typedef Eigen::Map<Eigen::VectorXi> MapVectorXi;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> dSparseMatrix;
typedef Eigen::SparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::SparseMatrix<real, Eigen::RowMajor, std::ptrdiff_t> realSparseMatrix;
typedef Eigen::Triplet<int> Triplet;

int TRIPLET_VECTOR_SIZE = 10000000;


// [[Rcpp::export]]
Rcpp::List EigenwordsRedSVD(
  MapVectorXi& sentence, int window_size, int vocab_size,
  int k, bool mode_oscca) {
  
  unsigned long long i, j, j2;
  unsigned long long sentence_size = sentence.size();
  unsigned long long lr_col_size = (unsigned long long)window_size*(unsigned long long)vocab_size;
  unsigned long long c_col_size = 2*lr_col_size;
  unsigned long long n_pushed_triplets = 0;
  long long i_word1, i_word2;
  int i_offset1, i_offset2;
  int offsets[2*window_size];
  
  iSparseMatrix tWC(vocab_size, c_col_size), tWC_temp(vocab_size, c_col_size);
  iSparseMatrix tLL(lr_col_size, lr_col_size), tLL_temp(lr_col_size, lr_col_size);
  iSparseMatrix tLR(lr_col_size, lr_col_size), tLR_temp(lr_col_size, lr_col_size);
  iSparseMatrix tRR(lr_col_size, lr_col_size), tRR_temp(lr_col_size, lr_col_size);
  MatrixXreal phi_l, phi_r;
  
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  tWW_diag.setZero();
  tCC_diag.setZero();
  
  std::vector<Triplet> tWC_tripletList;
  std::vector<Triplet> tLL_tripletList;
  std::vector<Triplet> tLR_tripletList;
  std::vector<Triplet> tRR_tripletList;
  tWC_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tLL_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tLR_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tRR_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  
  
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
  
  // Construct offset table (ex. [-2, -1, 1, 2])
  i_offset1 = 0;
  for (int offset=-window_size; offset<=window_size; offset++){
    if (offset != 0) {
      offsets[i_offset1] = offset;
      i_offset1++;
    }
  }
  
  for (unsigned long long i_sentence=0; i_sentence<sentence_size; i_sentence++) {
    
    i = sentence[i_sentence];
    tWW_diag(i) += 1;
    
    for (i_offset1=0; i_offset1<2*window_size; i_offset1++) {
      i_word1 = i_sentence + offsets[i_offset1];
      if (i_word1 >= 0 && i_word1 < sentence_size) {
        j = sentence[i_word1] + vocab_size * i_offset1;
        
        if (mode_oscca) {
          // One Step CCA
          tCC_diag(j) += 1;
          
        } else {
          // Two step CCA
          for (i_offset2=0; i_offset2<2*window_size; i_offset2++) {
            i_word2 = i_sentence + offsets[i_offset2];
            
            if (i_word2 >= 0 && i_word2 < sentence_size) {
              j2 = sentence[i_word2] + vocab_size * i_offset2;
              
              if (j < lr_col_size) {
                if (j2 < lr_col_size) {
                  // Upper left block of Ccc
                  if (j <= j2) {
                    // (j, j2) is upper-triangular part of tLL
                    tLL_tripletList.push_back(Triplet(j, j2, 1));
                  }
                } else {
                  // Upper right block of Ccc
                  tLR_tripletList.push_back(Triplet(j, j2 - lr_col_size, 1));
                }
              } else {
                if (j2 >= lr_col_size) {
                  // Lower right block of Ccc
                  if (j <= j2) {
                    // (j, j2) is upper-triangular part of tRR
                    tRR_tripletList.push_back(Triplet(j - lr_col_size, j2 - lr_col_size, 1));
                  }
                }
              }
            }
          }
        }
        
        tWC_tripletList.push_back(Triplet(i, j, 1));
      }
    }
    
    n_pushed_triplets++;
    
    // Commit temporary matrices
    if (n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size || i_sentence == sentence_size - 1) {
      tWC_temp.setFromTriplets(tWC_tripletList.begin(), tWC_tripletList.end());
      tWC_tripletList.clear();
      tWC += tWC_temp;
      tWC_temp.setZero();
      
      if (!mode_oscca) {
        tLL_temp.setFromTriplets(tLL_tripletList.begin(), tLL_tripletList.end());
        tLL_tripletList.clear();
        tLL += tLL_temp;
        tLL_temp.setZero();
        
        tLR_temp.setFromTriplets(tLR_tripletList.begin(), tLR_tripletList.end());
        tLR_tripletList.clear();
        tLR += tLR_temp;
        tLR_temp.setZero();
        
        tRR_temp.setFromTriplets(tRR_tripletList.begin(), tRR_tripletList.end());
        tRR_tripletList.clear();
        tRR += tRR_temp;
        tRR_temp.setZero();
      }
      
      n_pushed_triplets = 0;
    }
  }
  
  
  if (mode_oscca) {
    // Execute One Step CCA
    
    std::cout << "Calculate OSCCA..." << std::endl;
    std::cout << "Density of tWC = " << tWC.nonZeros() << "/" << tWC.rows() * tWC.cols() << std::endl;
    
    VectorXreal tWW_h(tWW_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
    VectorXreal tCC_h(tCC_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
    realSparseMatrix tWW_h_diag(tWW_h.size(), tWW_h.size());
    realSparseMatrix tCC_h_diag(tCC_h.size(), tCC_h.size());

    for (int ii = 0; ii<tWW_h.size(); ii++) {
      tWW_h_diag.insert(ii, ii) = tWW_h(ii);
    }
    
    for (int ii = 0; ii<tCC_h.size(); ii++) {
      tCC_h_diag.insert(ii, ii) = tCC_h(ii);
    }

    realSparseMatrix a(tWW_h_diag * (tWC.cast <real> ().eval().cwiseSqrt()) * tCC_h_diag);
    
    std::cout << "Calculate Randomized SVD..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdA(a, k, 20);
    
    return Rcpp::List::create(
//      Rcpp::Named("tWC") = Rcpp::wrap(tWC.cast <real> ()),
      Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
//      Rcpp::Named("tCC_h") = Rcpp::wrap(tCC_h),
//      Rcpp::Named("A") = Rcpp::wrap(a),
//      Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
      Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
      Rcpp::Named("word_vector") = Rcpp::wrap(tWW_h_diag * svdA.matrixU()),
      Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
      Rcpp::Named("k") = Rcpp::wrap(k)
      );
      
  } else {
    // Execute Two Step CCA
    
    tLL.makeCompressed();
    tLR.makeCompressed();
    tRR.makeCompressed();
    
    std::cout << "Density of tWC = " << tWC.nonZeros() << "/" << tWC.rows() << "*" << tWC.cols() << std::endl;
    std::cout << "Density of tLL = " << tLL.nonZeros() << "/" << tLL.rows() << "*" <<  tLL.cols() << std::endl;
    std::cout << "Density of tLR = " << tLR.nonZeros() << "/" << tLR.rows() << "*" <<  tLR.cols() << std::endl;
    std::cout << "Density of tRR = " << tRR.nonZeros() << "/" << tRR.rows() << "*" <<  tRR.cols() << std::endl;
    std::cout << "Calculate TSCCA..." << std::endl;
    
    // Two Step CCA : Step 1
    VectorXreal tLL_h(tLL.diagonal().cast <real> ().cwiseInverse().cwiseSqrt());
    VectorXreal tRR_h(tRR.diagonal().cast <real> ().cwiseInverse().cwiseSqrt());
    realSparseMatrix b(tLL_h.asDiagonal() * (tLR.cast <real> ().eval()) * tRR_h.asDiagonal());
    std::cout << "Density of b = " << b.nonZeros() << "/" << b.rows() * b.cols() << std::endl;
    
    std::cout << "Calculate Randomized SVD (1/2)..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdB(b, k, 20);
    b.resize(0, 0);
    
    // Two Step CCA : Step 2
    phi_l = svdB.matrixU();
    phi_r = svdB.matrixV();
    
    VectorXreal tWW_h(tWW_diag.cast <real> ().cwiseInverse().cwiseSqrt());
    VectorXreal tSS_h1((phi_l.transpose() * (tLL.cast <real> ().selfadjointView<Eigen::Upper>() * phi_l)).eval().diagonal().cwiseInverse().cwiseSqrt());
    VectorXreal tSS_h2((phi_r.transpose() * (tRR.cast <real> ().selfadjointView<Eigen::Upper>() * phi_r)).eval().diagonal().cwiseInverse().cwiseSqrt());
    
    tLL.resize(0, 0);
    tRR.resize(0, 0);
    
    VectorXreal tSS_h(2*window_size*k);
    
    std::cout << phi_l.rows() << " " << phi_l.cols() << std::endl;
    std::cout << phi_r.rows() << " " << phi_r.cols() << std::endl;
    std::cout << tSS_h1.rows() << " " << tSS_h1.cols() << std::endl;
    std::cout << tSS_h2.rows() << " " << tSS_h2.cols() << std::endl;
    
    tSS_h << tSS_h1, tSS_h2;
    MatrixXreal tWS(vocab_size, 2*window_size*k);
    tWS << tWC.topLeftCorner(vocab_size, c_col_size/2).cast <real> () * phi_l, tWC.topRightCorner(vocab_size, c_col_size/2).cast <real> () * phi_r;
    MatrixXreal a(tWW_h.asDiagonal() * tWS * tSS_h.asDiagonal());
    
    std::cout << "Calculate Randomized SVD (2/2)..." << std::endl;
    RedSVD::RedSVD<MatrixXreal> svdA(a, k, 20);
    
    return Rcpp::List::create(
//      Rcpp::Named("tWS") = Rcpp::wrap(tWS),
//      Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
//      Rcpp::Named("tSS_h") = Rcpp::wrap(tSS_h),
//      Rcpp::Named("A") = Rcpp::wrap(a),
//      Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
      Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
      Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
      Rcpp::Named("k") = Rcpp::wrap(k)
      );
  }
}
