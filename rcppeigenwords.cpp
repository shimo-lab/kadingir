/*
 * rcppeigenwords.cpp
 *
 * memo :
 *  - sentence の要素で 0 となっている要素は <OOV> (Out of Vocabulary, vocabulary に入っていない単語) に対応する．
 *  - v.asDiagonal() は疎行列ではなく密行列を返すため，仕方なく同様の処理をベタ書きしている箇所がある．
 *  - tWC のような表記は，行列 W, C の crossprod (Rでいうところの t(W) %*% C) を表す．
 *  - `_h` in `tWW_h` means "cast, diagonal, cwiseInverse, cwizeSqrt, cwiseSqrt"
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
                              iSparseMatrix &tXX_temp, iSparseMatrix &tXX)
{
  tXX_temp.setFromTriplets(tXX_tripletList.begin(), tXX_tripletList.end());
  tXX_tripletList.clear();
  tXX += tXX_temp;
  tXX_temp.setZero();
}

void fill_offset_table (int offsets[], int window_size)
{
  int i_offset1 = 0;
  for (int offset = -window_size; offset <= window_size; offset++){
    if (offset != 0) {
      offsets[i_offset1] = offset;
      i_offset1++;
    }
  }
}

void construct_crossprod_matrices (const MapVectorXi& sentence,
                                   Eigen::VectorXi &tWW_diag, Eigen::VectorXi &tCC_diag,
                                   iSparseMatrix &tWC, iSparseMatrix &tLL,
                                   iSparseMatrix &tLR, iSparseMatrix &tRR,
                                   const int window_size, const int vocab_size,
                                   const bool mode_oscca)
{
  const unsigned long long lr_col_size = (unsigned long long)window_size * vocab_size;
  const unsigned long long c_col_size = 2 * lr_col_size;
  const unsigned long long sentence_size = sentence.size();
  unsigned long long n_pushed_triplets = 0;

  std::vector<Triplet> tWC_tripletList, tLL_tripletList, tLR_tripletList, tRR_tripletList;
  tWC_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tLL_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tLR_tripletList.reserve(TRIPLET_VECTOR_SIZE);
  tRR_tripletList.reserve(TRIPLET_VECTOR_SIZE);

  iSparseMatrix tWC_temp(vocab_size, c_col_size);
  iSparseMatrix tLL_temp(lr_col_size, lr_col_size);
  iSparseMatrix tLR_temp(lr_col_size, lr_col_size);
  iSparseMatrix tRR_temp(lr_col_size, lr_col_size);

  tWW_diag.setZero();
  tCC_diag.setZero();

  // Construct offset table (If window_size=2, offsets = [-2, -1, 1, 2])
  int offsets[2*window_size];
  fill_offset_table(offsets, window_size);
  
  
  for (unsigned long long i_sentence = 0; i_sentence < sentence_size; i_sentence++) {
    unsigned long long i = sentence[i_sentence];
    tWW_diag(i) += 1;
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      long long i_word1 = i_sentence + offsets[i_offset1];
      
      // If `i_word1` is out of indices of sentence
      if ((i_word1 < 0) || (i_word1 >= sentence_size)) continue;
      
      unsigned long long word1 = sentence[i_word1] + vocab_size * i_offset1;
      
      if (mode_oscca) {
        // One Step CCA
        tCC_diag(word1) += 1;
        
      } else {
        // Two step CCA
        for (int i_offset2 = 0; i_offset2 < 2 * window_size; i_offset2++) {
          long long i_word2 = i_sentence + offsets[i_offset2];
          
          // If `i_word2` is out of indices of sentence
          if ((i_word2 < 0) || (i_word2 >= sentence_size)) continue;
          
          unsigned long long word2 = sentence[i_word2] + vocab_size * i_offset2;
          
          bool word1_in_left_context = word1 < lr_col_size;
          bool word2_in_left_context = word2 < lr_col_size;
          bool is_upper_triangular = word1 <= word2;
          
          if (word1_in_left_context && word2_in_left_context && is_upper_triangular) {
            // (word1, word2) is an element of upper-triangular part of tLL
            tLL_tripletList.push_back(Triplet(word1, word2, 1));
          } else if (word1_in_left_context && !word2_in_left_context) {
            // (word1, word2) is an element of tLR
            tLR_tripletList.push_back(Triplet(word1, word2 - lr_col_size, 1));
          } else if (!word1_in_left_context && !word2_in_left_context && is_upper_triangular) {
            // (word1, word2) is an element of upper-triangular part of tRR
            tRR_tripletList.push_back(Triplet(word1 - lr_col_size, word2 - lr_col_size, 1));
          }
        }
      }
      
      tWC_tripletList.push_back(Triplet(i, word1, 1));
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
}


void construct_crossprod_matrices_documents (const MapVectorXi& sentence, const MapVectorXi& document_id,
                                             Eigen::VectorXi &tWW_diag, Eigen::VectorXi &tCC_diag,
                                             Eigen::VectorXi &tDD_diag, iSparseMatrix &H,
                                             const int window_size, const int vocab_size)
{
  const unsigned long long sentence_size = sentence.size();
  const unsigned long long c_col_size = 2 * (unsigned long long)window_size * vocab_size;
  const unsigned long long n_documents = document_id.maxCoeff() + 1;
  const unsigned long long p_cumsum[4] = {1, vocab_size, vocab_size + c_col_size, vocab_size + c_col_size + n_documents};

  unsigned long long n_pushed_triplets = 0;

  std::vector<Triplet> H_tripletList;
  H_tripletList.reserve(TRIPLET_VECTOR_SIZE);

  iSparseMatrix H_temp(p_cumsum[3], p_cumsum[3]);

  tWW_diag.setZero();
  tCC_diag.setZero();
  tDD_diag.setZero();

  // Construct offset table (If window_size=2, offsets = [-2, -1, 1, 2])
  int offsets[2*window_size];
  fill_offset_table(offsets, window_size);
    
  for (unsigned long long i_sentence = 0; i_sentence < sentence_size; i_sentence++) {
    unsigned long long w_id = sentence[i_sentence];
    unsigned long long d_id = document_id[i_sentence];

    tWW_diag(w_id) += 1;
    tDD_diag(d_id) += 1;
    H_tripletList.push_back(Triplet(p_cumsum[0] + w_id - 1,  p_cumsum[2] + d_id - 1,  1));  // Element of tWD
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      long long i_word1 = i_sentence + offsets[i_offset1];
      
      // If `i_word1` is out of indices of sentence
      if ((i_word1 < 0) || (i_word1 >= sentence_size)) continue;
      
      unsigned long long word1 = sentence[i_word1] + vocab_size * i_offset1;

      tCC_diag(word1) += 1;

      H_tripletList.push_back(Triplet(p_cumsum[0] + w_id  - 1, p_cumsum[1] + word1 - 1, 1));  // Element of tWC
      H_tripletList.push_back(Triplet(p_cumsum[1] + word1 - 1, p_cumsum[2] + d_id  - 1, 1));  // Element of tCD
    }
    
    n_pushed_triplets += 2*window_size + 1;
    
    // Commit temporary matrices
    if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_sentence == sentence_size - 1)) {
      update_crossprod_matrix(H_tripletList, H_temp, H);

      n_pushed_triplets = 0;
    }
  }

  H.makeCompressed();

  std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
  std::cout << "H,  " << H.nonZeros() << ",  " << H.rows() << ",  " << H.cols() << std::endl;
  std::cout << std::endl;
}


void construct_h_diag_matrix (Eigen::VectorXi &tXX_diag, realSparseMatrix &tXX_h_diag)
{
  VectorXreal tXX_h(tXX_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  
  for (int i = 0; i < tXX_h.size(); i++) {
    tXX_h_diag.insert(i, i) = tXX_h(i);
  }
}


// [[Rcpp::export]]
Rcpp::List EigenwordsRedSVD(const MapVectorXi& sentence, const int window_size,
                            const int vocab_size, const int k, const bool mode_oscca)
{
  const unsigned long long lr_col_size = (unsigned long long)window_size * vocab_size;
  const unsigned long long c_col_size = 2 * lr_col_size;
  
  
  // Construct crossprod matrices
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  iSparseMatrix tWC(vocab_size, c_col_size);
  iSparseMatrix tLL(lr_col_size, lr_col_size);
  iSparseMatrix tLR(lr_col_size, lr_col_size);
  iSparseMatrix tRR(lr_col_size, lr_col_size);

  construct_crossprod_matrices(sentence, tWW_diag, tCC_diag,
                               tWC, tLL, tLR, tRR,
                               window_size, vocab_size, mode_oscca);


  // Construct the matrices for CCA and execute CCA
  realSparseMatrix tWW_h_diag(vocab_size, vocab_size);
  construct_h_diag_matrix(tWW_diag, tWW_h_diag);

  if (mode_oscca) {
    // Execute One Step CCA
    std::cout << "Calculate OSCCA..." << std::endl;
    
    realSparseMatrix tCC_h_diag(c_col_size, c_col_size);
    construct_h_diag_matrix(tCC_diag, tCC_h_diag);
    
    realSparseMatrix a = tWW_h_diag * (tWC.cast <real> ().eval().cwiseSqrt()) * tCC_h_diag;
   
    std::cout << "Calculate Randomized SVD..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdA(a, k, 20);
    
    return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(tWW_h_diag * svdA.matrixU()),
                              // Rcpp::Named("tWC") = Rcpp::wrap(tWC.cast <real> ()),
                              // Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
                              // Rcpp::Named("tCC_h") = Rcpp::wrap(tCC_h),
                              // Rcpp::Named("A") = Rcpp::wrap(a),
                              // Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
                              // Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
                               Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues())
                              );

  } else {
    // Execute Two Step CCA
    std::cout << "Calculate TSCCA..." << std::endl;
    
    // Two Step CCA : Step 1
    Eigen::VectorXi tLL_diag = tLL.diagonal();
    Eigen::VectorXi tRR_diag = tRR.diagonal();
    realSparseMatrix tLL_h_diag(lr_col_size, lr_col_size);
    realSparseMatrix tRR_h_diag(lr_col_size, lr_col_size);
    
    construct_h_diag_matrix(tLL_diag, tLL_h_diag);
    construct_h_diag_matrix(tRR_diag, tRR_h_diag);
    realSparseMatrix b = tLL_h_diag * (tLR.cast <real> ().eval().cwiseSqrt()) * tRR_h_diag;
    
    std::cout << "# of nonzero,  # of rows,  # of cols = " << b.nonZeros() << ",  " << b.rows() << ",  " << b.cols() << std::endl;
    
    std::cout << "Calculate Randomized SVD (1/2)..." << std::endl;
    RedSVD::RedSVD<realSparseMatrix> svdB(b, k, 20);
    b.resize(0, 0);  // Release memory
    
    MatrixXreal phi_l = svdB.matrixU();
    MatrixXreal phi_r = svdB.matrixV();
    
    // Two Step CCA : Step 2
    VectorXreal tSS_h1 = (phi_l.transpose() * (tLL.cast <real> ().selfadjointView<Eigen::Upper>() * phi_l)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt();
    VectorXreal tSS_h2 = (phi_r.transpose() * (tRR.cast <real> ().selfadjointView<Eigen::Upper>() * phi_r)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt();
    
    // Release memory
    tLL.resize(0, 0);
    tRR.resize(0, 0);
    
    VectorXreal tSS_h(2*k);
    tSS_h << tSS_h1, tSS_h2;
    realSparseMatrix tSS_h_diag(tSS_h.size(), tSS_h.size());
    for (int i = 0; i < tSS_h.size(); i++) {
      tSS_h_diag.insert(i, i) = tSS_h(i);
    }

    MatrixXreal tWS(vocab_size, 2*k);
    tWS << tWC.topLeftCorner(vocab_size, lr_col_size).cast <real> ().cwiseSqrt() * phi_l, tWC.topRightCorner(vocab_size, lr_col_size).cast <real> ().cwiseSqrt() * phi_r;

    MatrixXreal a = tWW_h_diag * tWS * tSS_h_diag;
    
    std::cout << "Calculate Randomized SVD (2/2)..." << std::endl;
    RedSVD::RedSVD<MatrixXreal> svdA(a, k, 20);
    
    return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(tWW_h_diag * svdA.matrixU()),
                              // Rcpp::Named("tWS") = Rcpp::wrap(tWS),
                              // Rcpp::Named("tWW_h") = Rcpp::wrap(tWW_h),
                              // Rcpp::Named("tSS_h") = Rcpp::wrap(tSS_h),
                              // Rcpp::Named("A") = Rcpp::wrap(a),
                              // Rcpp::Named("V") = Rcpp::wrap(svdA.matrixV()),
                              // Rcpp::Named("U") = Rcpp::wrap(svdA.matrixU()),
                               Rcpp::Named("D") = Rcpp::wrap(svdA.singularValues())
                              );
  }
}


// [[Rcpp::export]]
Rcpp::List EigendocsRedSVD(const MapVectorXi& sentence, const MapVectorXi& document_id, const int window_size,
                           const int vocab_size, const int k, const bool mode_oscca,
                           const real gamma_G, const real gamma_H)
{
  const unsigned long long lr_col_size = (unsigned long long)window_size * vocab_size;
  const unsigned long long c_col_size = 2 * lr_col_size;
  const unsigned long long n_documents = document_id.maxCoeff() + 1;
  const unsigned long long p[4] = {0, vocab_size, c_col_size, n_documents};
  const unsigned long long p_cumsum[4] = {1, vocab_size, vocab_size + c_col_size, vocab_size + c_col_size + n_documents};
  const unsigned long long p_sum = vocab_size + c_col_size + n_documents;


  // Construct crossprod matrices
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  Eigen::VectorXi tDD_diag(n_documents);
  iSparseMatrix H(p_sum, p_sum);

  construct_crossprod_matrices_documents (sentence, document_id,
                                          tWW_diag, tCC_diag, tDD_diag, H,
                                          window_size, vocab_size);


  // Construct the matrices for CCA and execute CCA
  std::cout << "Calculate OSCCA..." << std::endl;
  
  Eigen::VectorXi G_diag(p_sum);
  G_diag << tWW_diag, tCC_diag, tDD_diag;
  G_diag += gamma_G * Eigen::VectorXi::Ones(p_sum);  // Regularization for G

  realSparseMatrix G_inv_sqrt(p_sum, p_sum);
  construct_h_diag_matrix(G_diag, G_inv_sqrt);
  G_inv_sqrt /= sqrt(2);

  realSparseMatrix H_reg(p_sum, p_sum);
  H_reg.setIdentity();
  H_reg *= gamma_H;

  realSparseMatrix A = (G_inv_sqrt * ((H.cast <real> ().cwiseSqrt() + H_reg).selfadjointView<Eigen::Upper>()) * G_inv_sqrt).eval();

  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<realSparseMatrix> svdA(A, k, 20);
  MatrixXreal principal_components = svdA.matrixV();
  MatrixXreal word_vector     = G_inv_sqrt.block(p_cumsum[0] - 1, p_cumsum[0] - 1, p[1], p[1]) * principal_components.block(p_cumsum[0] - 1, p_cumsum[0] - 1, p[1], k);
  MatrixXreal document_vector = G_inv_sqrt.block(p_cumsum[2] - 1, p_cumsum[2] - 1, p[3], p[3]) * principal_components.block(p_cumsum[2] - 1, p_cumsum[0] - 1, p[2], k);
  
  return Rcpp::List::create(Rcpp::Named("word_vector") = Rcpp::wrap(word_vector),
                            Rcpp::Named("document_vector") = Rcpp::wrap(document_vector));
}
