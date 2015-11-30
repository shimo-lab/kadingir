/*
 * rcppeigenwords.cpp
 *
 * memo :
 *  - sentence の要素で 0 となっている要素は <OOV> (Out of Vocabulary, vocabulary に入っていない単語) に対応する．
 *  - v.asDiagonal() は疎行列ではなく密行列を返すため，仕方なく同様の処理をベタ書きしている箇所がある．
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
typedef Eigen::SparseMatrix<int, Eigen::RowMajor, std::ptrdiff_t> iSparseMatrix;
typedef Eigen::SparseMatrix<real, Eigen::RowMajor, std::ptrdiff_t> realSparseMatrix;
typedef Eigen::Triplet<int> Triplet;

const int TRIPLET_VECTOR_SIZE = 10000000;


// Update crossprod matrix using triplets
void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
			 iSparseMatrix &tXX_temp, iSparseMatrix &tXX) {

  tXX_temp.setFromTriplets(tXX_tripletList.begin(), tXX_tripletList.end());
  tXX_tripletList.clear();
  tXX += tXX_temp;
  tXX_temp.setZero();
}

// [[Rcpp::export]]
Rcpp::List EigenwordsRedSVD(const MapVectorXi& sentence, const int window_size,
			    const int vocab_size, const int k, const bool mode_oscca) {
  
  const unsigned long long sentence_size = sentence.size();
  const unsigned long long lr_col_size = (unsigned long long)window_size*(unsigned long long)vocab_size;
  const unsigned long long c_col_size = 2*lr_col_size;
  
  iSparseMatrix tWC(vocab_size, c_col_size), tWC_temp(vocab_size, c_col_size);
  iSparseMatrix tLL(lr_col_size, lr_col_size), tLL_temp(lr_col_size, lr_col_size);
  iSparseMatrix tLR(lr_col_size, lr_col_size), tLR_temp(lr_col_size, lr_col_size);
  iSparseMatrix tRR(lr_col_size, lr_col_size), tRR_temp(lr_col_size, lr_col_size);
  MatrixXreal phi_l, phi_r;
  
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  tWW_diag.setZero();
  tCC_diag.setZero();
  
  std::vector<Triplet> tWC_tripletList, tLL_tripletList, tLR_tripletList, tRR_tripletList;
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
  std::cout << "dim of output = " << k             << std::endl;
  std::cout << "sentence size = " << sentence_size << std::endl;
  std::cout << "c_col_size    = " << c_col_size    << std::endl;
  std::cout << std::endl;
  
  // Construct offset table (If window_size=2, offsets = [-2, -1, 1, 2])
  int offsets[2*window_size];

  {
    int i_offset1 = 0;
    for (int offset = -window_size; offset <= window_size; offset++){
      if (offset != 0) {
	offsets[i_offset1] = offset;
	i_offset1++;
      }
    }
  }
  
  // Construct crossprod matrices
  unsigned long long n_pushed_triplets = 0;

  for (unsigned long long i_sentence = 0; i_sentence < sentence_size; i_sentence++) {
    
    unsigned long long i = sentence[i_sentence];
    tWW_diag(i) += 1;
    
    for (int i_offset1 = 0; i_offset1 < 2*window_size; i_offset1++) {
      long long i_word1 = i_sentence + offsets[i_offset1];
      if ((i_word1 >= 0) && (i_word1 < sentence_size)) {
        unsigned long long j = sentence[i_word1] + vocab_size * i_offset1;
        
        if (mode_oscca) {
          // One Step CCA
          tCC_diag(j) += 1;

        } else {
          // Two step CCA
          for (int i_offset2 = 0; i_offset2 < 2*window_size; i_offset2++) {
            long long i_word2 = i_sentence + offsets[i_offset2];
            
            if ((i_word2 >= 0) && (i_word2 < sentence_size)) {
              unsigned long long j2 = sentence[i_word2] + vocab_size * i_offset2;
              
	      if ((j < lr_col_size) && (j2 < lr_col_size) && (j <= j2)) {
		// (j, j2) is an element of upper-triangular part of tLL
		tLL_tripletList.push_back(Triplet(j, j2, 1));
	      }
	      if ((j < lr_col_size) && (j2 >= lr_col_size)) {
		// (j, j2) is an element of tLR
		tLR_tripletList.push_back(Triplet(j, j2 - lr_col_size, 1));
	      }
	      if ((j >= lr_col_size) && (j2 >= lr_col_size) && (j <= j2)) {
		// (j, j2) is an element of upper-triangular part of tRR
		tRR_tripletList.push_back(Triplet(j - lr_col_size, j2 - lr_col_size, 1));
	      }
            }
          }
        }
        
        tWC_tripletList.push_back(Triplet(i, j, 1));
      }
    }
    
    n_pushed_triplets++;
    
    // Commit temporary matrices
    if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_sentence == sentence_size - 1)) {
      update_crossprod_matrix(tWC_tripletList, tWC_temp, tWC);
            
      if (!mode_oscca) {
        update_crossprod_matrix(tLL_tripletList, tLL_temp, tLL);
        update_crossprod_matrix(tLR_tripletList, tLR_temp, tLR);
        update_crossprod_matrix(tRR_tripletList, tRR_temp, tRR);
       }

       n_pushed_triplets = 0;
     }
   }

   tWC.makeCompressed();
   tLL.makeCompressed();
   tLR.makeCompressed();
   tRR.makeCompressed();

   std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
   std::cout << "tWC,  " << tWC.nonZeros() << ",  " << tWC.rows() << ",  " << tWC.cols() << std::endl;
   std::cout << "tLL,  " << tLL.nonZeros() << ",  " << tLL.rows() << ",  " << tLL.cols() << std::endl;
   std::cout << "tLR,  " << tLR.nonZeros() << ",  " << tLR.rows() << ",  " << tLR.cols() << std::endl;
   std::cout << "tRR,  " << tRR.nonZeros() << ",  " << tRR.rows() << ",  " << tRR.cols() << std::endl;
   std::cout << std::endl;


   // `_h` in `tWW_h` means "cast, diagonal, cwiseInverse, cwizeSqrt, cwiseSqrt"
   VectorXreal tWW_h(tWW_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
   VectorXreal tCC_h(tCC_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
   realSparseMatrix tWW_h_diag(tWW_h.size(), tWW_h.size());
   realSparseMatrix tCC_h_diag(tCC_h.size(), tCC_h.size());

   for (int i = 0; i < tWW_h.size(); i++) {
     tWW_h_diag.insert(i, i) = tWW_h(i);
   }
   if (mode_oscca) {
     for (int i = 0; i < tCC_h.size(); i++) {
       tCC_h_diag.insert(i, i) = tCC_h(i);
     }
   }


   // Construct the matrices for CCA and execute CCA
   if (mode_oscca) {
     // Execute One Step CCA
    std::cout << "Calculate OSCCA..." << std::endl;
    
    realSparseMatrix a(tWW_h_diag * (tWC.cast <real> ().eval().cwiseSqrt()) * tCC_h_diag);
   
    std::cout << "Calculate Randomized SVD..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdA(a, k, 20);
    
    return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(tWW_h_diag * svdA.matrixU()),
			      // Rcpp::Named("tWC") = Rcpp::wrap(tWC.cast <real> ()),
			      // Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
			      // Rcpp::Named("tCC_h") = Rcpp::wrap(tCC_h),
			      // Rcpp::Named("A") = Rcpp::wrap(a),
			      // Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
			      // Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
			      // Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
			      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
			      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
			      Rcpp::Named("k") = Rcpp::wrap(k)
			      );
      
  } else {
    // Execute Two Step CCA
    std::cout << "Calculate TSCCA..." << std::endl;
    
    // Two Step CCA : Step 1
    VectorXreal tLL_h(tLL.diagonal().cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
    VectorXreal tRR_h(tRR.diagonal().cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
    realSparseMatrix b(tLL_h.asDiagonal() * (tLR.cast <real> ().eval().cwiseSqrt()) * tRR_h.asDiagonal());
    std::cout << "# of nonzero,  # of rows,  # of cols = " << b.nonZeros() << ",  " << b.rows() << ",  " << b.cols() << std::endl;
    
    std::cout << "Calculate Randomized SVD (1/2)..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdB(b, k, 20);
    b.resize(0, 0);  // Release memory

    phi_l = svdB.matrixU();
    phi_r = svdB.matrixV();
    
    // Two Step CCA : Step 2
    VectorXreal tSS_h1((phi_l.transpose() * (tLL.cast <real> ().selfadjointView<Eigen::Upper>() * phi_l)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt());
    VectorXreal tSS_h2((phi_r.transpose() * (tRR.cast <real> ().selfadjointView<Eigen::Upper>() * phi_r)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt());
    
    // Release memory
    tLL.resize(0, 0);
    tRR.resize(0, 0);
    
    VectorXreal tSS_h(2*window_size*k);
    tSS_h << tSS_h1, tSS_h2;
    realSparseMatrix tSS_h_diag(tSS_h.size(), tSS_h.size());
    for (int ii = 0; ii<tSS_h.size(); ii++) {
      tSS_h_diag.insert(ii, ii) = tSS_h(ii);
    }

    MatrixXreal tWS(vocab_size, 2*window_size*k);
    tWS << tWC.topLeftCorner(vocab_size, lr_col_size).cast <real> ().cwiseSqrt() * phi_l, tWC.topRightCorner(vocab_size, lr_col_size).cast <real> ().cwiseSqrt() * phi_r;

    MatrixXreal a(tWW_h_diag * tWS * tSS_h_diag);
    
    std::cout << "Calculate Randomized SVD (2/2)..." << std::endl;
    RedSVD::RedSVD<MatrixXreal> svdA(a, k, 20);
    
    return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(tWW_h_diag * svdA.matrixU()),
			      // Rcpp::Named("tWS") = Rcpp::wrap(tWS),
			      // Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
			      // Rcpp::Named("tSS_h") = Rcpp::wrap(tSS_h),
			      // Rcpp::Named("A") = Rcpp::wrap(a),
			      // Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
			      // Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
			      // Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues()),
			      Rcpp::Named("window.size") = Rcpp::wrap(window_size),
			      Rcpp::Named("vocab.size") = Rcpp::wrap(vocab_size),
			      Rcpp::Named("k") = Rcpp::wrap(k)
			      );
  }
}
