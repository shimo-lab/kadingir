/*
 * kadingir_core.cpp
 *
 * memo :
 *  - sentence の要素で 0 となっている要素は <OOV> (Out of Vocabulary, vocabulary に入っていない単語) に対応する．
 *  - v.asDiagonal() は疎行列ではなく密行列を返すため，仕方なく同様の処理をベタ書きしている箇所がある．
 *  - tWC のような表記は，行列 W, C の crossprod (Rでいうところの t(W) %*% C) を表す．
 *  - `_h` in `tWW_h` means "cast, diagonal, cwiseInverse, cwizeSqrt, cwiseSqrt"
 */


#include "kadingir_core.hpp"

const int TRIPLET_VECTOR_SIZE = 10000000;


// Update crossprod matrix using triplets
template <class MatrixX> void update_crossprod_matrix (std::vector<Triplet> &tXX_tripletList,
                                                       MatrixX &tXX_temp,
                                                       MatrixX &tXX)
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


void construct_h_diag_matrix (Eigen::VectorXi &tXX_diag, realSparseMatrix &tXX_h_diag)
{
  VectorXreal tXX_h(tXX_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  tXX_h_diag.reserve(tXX_diag.size());
  
  for (int i = 0; i < tXX_h.size(); i++) {
    tXX_h_diag.insert(i, i) = tXX_h(i);
  }
}

// TODO : template で書けないか？
void construct_h_diag_matrix (VectorXreal &tXX_diag, realSparseMatrix &tXX_h_diag)
{
  VectorXreal tXX_h(tXX_diag.cast <real> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  tXX_h_diag.reserve(tXX_diag.size());
  
  for (int i = 0; i < tXX_h.size(); i++) {
    tXX_h_diag.insert(i, i) = tXX_h(i);
  }
}


Eigenwords::Eigenwords (const MapVectorXi& _sentence,
                        const int _window_size,
                        const int _vocab_size,
                        const int _k,
                        const bool _mode_oscca
                        ) : sentence(_sentence),
                            window_size(_window_size),
                            vocab_size(_vocab_size),
                            k(_k),
                            mode_oscca(_mode_oscca)
{
  lr_col_size = (unsigned long long)window_size * vocab_size;
  c_col_size = 2 * lr_col_size;
}

void Eigenwords::compute()
{
  // Construct crossprod matrices
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  iSparseMatrix tWC(vocab_size, c_col_size);
  iSparseMatrix tLL(lr_col_size, lr_col_size);
  iSparseMatrix tLR(lr_col_size, lr_col_size);
  iSparseMatrix tRR(lr_col_size, lr_col_size);

  construct_matrices(tWW_diag, tCC_diag, tWC, tLL, tLR, tRR);


  // Construct the matrices for CCA and execute CCA
  realSparseMatrix tWW_h_diag(vocab_size, vocab_size);
  construct_h_diag_matrix(tWW_diag, tWW_h_diag);

  if (mode_oscca) {
    run_oscca(tWW_h_diag, tWC, tCC_diag);
  } else {
    run_tscca(tWW_h_diag, tLL, tLR, tRR, tWC);
  }
}

MatrixXreal Eigenwords::get_word_vectors() {
  return word_vectors;
}

MatrixXreal Eigenwords::get_context_vectors() {
  return context_vectors;
}

VectorXreal Eigenwords::get_singular_values() {
  return singular_values;
}

void Eigenwords::construct_matrices (Eigen::VectorXi &tWW_diag,
                                     Eigen::VectorXi &tCC_diag,
                                     iSparseMatrix &tWC,
                                     iSparseMatrix &tLL,
                                     iSparseMatrix &tLR,
                                     iSparseMatrix &tRR)
{
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
    const unsigned long long word0 = sentence[i_sentence];
    tWW_diag(word0) += 1;
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      const long long i_word1 = i_sentence + offsets[i_offset1];
      
      // If `i_word1` is out of indices of sentence
      if ((i_word1 < 0) || (i_word1 >= sentence_size)) continue;
      
      const unsigned long long word1 = sentence[i_word1] + vocab_size * i_offset1;
      
      if (mode_oscca) {
        // One Step CCA
        tCC_diag(word1) += 1;
        
      } else {
        // Two step CCA
        for (int i_offset2 = 0; i_offset2 < 2 * window_size; i_offset2++) {
          const long long i_word2 = i_sentence + offsets[i_offset2];
          
          // If `i_word2` is out of indices of sentence
          if ((i_word2 < 0) || (i_word2 >= sentence_size)) continue;
          
          const unsigned long long word2 = sentence[i_word2] + vocab_size * i_offset2;
          
          const bool word1_in_left_context = word1 < lr_col_size;
          const bool word2_in_left_context = word2 < lr_col_size;
          const bool is_upper_triangular = word1 <= word2;
          
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
      
      tWC_tripletList.push_back(Triplet(word0, word1, 1));
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

// Execute One Step CCA
void Eigenwords::run_oscca(realSparseMatrix &tWW_h_diag,
                           iSparseMatrix &tWC,
                           Eigen::VectorXi &tCC_diag)
{
  std::cout << "Calculate OSCCA..." << std::endl;
  
  realSparseMatrix tCC_h_diag(c_col_size, c_col_size);
  construct_h_diag_matrix(tCC_diag, tCC_h_diag);
  
  realSparseMatrix a = tWW_h_diag * (tWC.cast <real> ().eval().cwiseSqrt()) * tCC_h_diag;
  
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<realSparseMatrix> svdA(a, k, 20);
  
  word_vectors = tWW_h_diag * svdA.matrixU();
  context_vectors = tCC_h_diag * svdA.matrixV();
  singular_values = svdA.singularValues();
}

// Execute Two Step CCA
void Eigenwords::run_tscca(realSparseMatrix &tWW_h_diag,
                           iSparseMatrix &tLL,
                           iSparseMatrix &tLR,
                           iSparseMatrix &tRR,
                           iSparseMatrix &tWC)
{
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

  word_vectors = tWW_h_diag * svdA.matrixU();
  context_vectors = tSS_h_diag * svdA.matrixV();
  singular_values = svdA.singularValues();
}




Eigendocs::Eigendocs (const MapVectorXi& _sentence,
                      const MapVectorXi& _document_id,
                      const int _window_size,
                      const int _vocab_size,
                      const int _k,
                      const bool _link_w_d,
                      const bool _link_c_d,
                      const real _gamma_G,
                      const real _gamma_H
                      ) : sentence(_sentence),
                          document_id(_document_id),
                          window_size(_window_size),
                          vocab_size(_vocab_size),
                          k(_k),
                          link_w_d(_link_w_d),
                          link_c_d(_link_c_d),
                          gamma_G(_gamma_G),
                          gamma_H(_gamma_H)
{
  lr_col_size = (unsigned long long)window_size * vocab_size;
  c_col_size = 2 * lr_col_size;

  p_head_domains[0] = 0;
  p_head_domains[1] = vocab_size;
  p_head_domains[2] = vocab_size + c_col_size;
}

void Eigendocs::compute()
{
  const unsigned long long n_documents = document_id.maxCoeff() + 1;

  p_indices[0] = vocab_size;
  p_indices[1] = c_col_size;
  p_indices[2] = n_documents;
  p = vocab_size + c_col_size + n_documents;


  // Construct crossprod matrices
  Eigen::VectorXi tWW_diag(vocab_size);
  Eigen::VectorXi tCC_diag(c_col_size);
  Eigen::VectorXi tDD_diag(n_documents);
  iSparseMatrix H(p, p);

  construct_matrices (tWW_diag, tCC_diag, tDD_diag, H);


  // Construct the matrices for CCA and execute CCA  
  Eigen::VectorXi G_diag(p);
  
  if (link_w_d && link_c_d) {
    G_diag << 2*tWW_diag, 2*tCC_diag, 2*tDD_diag;
  } else if (!link_w_d) {
    G_diag <<   tWW_diag, 2*tCC_diag,   tDD_diag;
  } else {
    G_diag << 2*tWW_diag,   tCC_diag,   tDD_diag;
  }

  realSparseMatrix G_inv_sqrt(p, p);
  construct_h_diag_matrix(G_diag, G_inv_sqrt);
  G_inv_sqrt /= sqrt(2);

  realSparseMatrix A = (G_inv_sqrt * ((H.cast <real> ().cwiseSqrt()).selfadjointView<Eigen::Upper>()) * G_inv_sqrt).eval();

  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<realSparseMatrix> svdA(A, k, 20);
  MatrixXreal principal_components = svdA.matrixV();
  vector_representations = G_inv_sqrt * principal_components.block(0, 0, p, k);
  singular_values = svdA.singularValues();
}

MatrixXreal Eigendocs::get_vector_representations() {
  return vector_representations;
}

VectorXreal Eigendocs::get_singular_values() {
  return singular_values;
}

void Eigendocs::construct_matrices (Eigen::VectorXi &tWW_diag,
                                    Eigen::VectorXi &tCC_diag,
                                    Eigen::VectorXi &tDD_diag,
                                    iSparseMatrix &H)
{
  const unsigned long long sentence_size = sentence.size();
  const unsigned long long n_documents = document_id.maxCoeff() + 1;

  unsigned long long n_pushed_triplets = 0;

  std::vector<Triplet> H_tripletList;
  H_tripletList.reserve(TRIPLET_VECTOR_SIZE);

  iSparseMatrix H_temp(p, p);

  tWW_diag.setZero();
  tCC_diag.setZero();
  tDD_diag.setZero();

  // Construct offset table (If window_size=2, offsets = [-2, -1, 1, 2])
  int offsets[2*window_size];
  fill_offset_table(offsets, window_size);
    
  for (unsigned long long i_sentence = 0; i_sentence < sentence_size; i_sentence++) {
    const unsigned long long word0 = sentence[i_sentence];
    const unsigned long long d_id = document_id[i_sentence];

    tWW_diag(word0) += 1;
    tDD_diag(d_id) += 1;
    
    if (link_w_d) {
      H_tripletList.push_back(Triplet(word0,  p_head_domains[2] + d_id,  1));  // Element of tWD
    }
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      const long long i_word1 = i_sentence + offsets[i_offset1];
      
      // If `i_word1` is out of indices of sentence
      if ((i_word1 < 0) || (i_word1 >= sentence_size)) continue;
      
      const unsigned long long word1 = sentence[i_word1] + vocab_size * i_offset1;

      tCC_diag(word1) += 1;

      H_tripletList.push_back(Triplet(word0, p_head_domains[1] + word1, 1));  // Element of tWC
      if (link_c_d) {
        H_tripletList.push_back(Triplet(p_head_domains[1] + word1, p_head_domains[2] + d_id, 1));  // Element of tCD
      }
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



MCEigendocs::MCEigendocs(const MapVectorXi& _sentence_concated,
                         const MapVectorXi& _document_id_concated,
                         const Eigen::VectorXi _window_sizes,
                         const Eigen::VectorXi _vocab_sizes,
                         const Eigen::VectorXi _sentence_lengths,
                         const int _k,
                         const real _gamma_G,
                         const real _gamma_H,
                         const bool _link_w_d,
                         const bool _link_c_d,
                         const bool _doc_weighting,
                         const real _weight_doc_vs_vc
                        ) : sentence_concated(_sentence_concated),
                        document_id_concated(_document_id_concated),
                        window_sizes(_window_sizes),
                        vocab_sizes(_vocab_sizes),
                        sentence_lengths(_sentence_lengths),
                        k(_k),
                        gamma_G(_gamma_G),
                        gamma_H(_gamma_H),
                        link_w_d(_link_w_d),
                        link_c_d(_link_c_d),
                        doc_weighting(_doc_weighting),
                        weight_doc_vs_vc(_weight_doc_vs_vc)
{
  if (window_sizes.size() != vocab_sizes.size()) {
    std::cout << "window_sizes.size() != vocab_sizes.size()" << std::endl;
  }
  n_languages = window_sizes.size();
  
  // dimension of context domain
  lr_col_sizes.resize(n_languages);
  c_col_sizes.resize(n_languages);
  for (int i = 0; i < n_languages; i++) {
    lr_col_sizes[i] = (unsigned long long)window_sizes(i) * vocab_sizes[i];
    c_col_sizes[i] = 2 * lr_col_sizes[i];
  }

  n_documents = document_id_concated.maxCoeff() + 1;
  n_domain = 2 * n_languages + 1;  // # of domains = 2 * (# of languages) + document
  p_indices.resize(n_domain);        // dimensions of each domain
  p_head_domains.resize(n_domain);   // head of indices of each domain

  for (int i = 0; i < n_domain - 1; i++) {
    int i_domain = i / 2;
    if (i % 2 == 0) {
      p_indices[i] = vocab_sizes[i_domain];
    } else {
      p_indices[i] = c_col_sizes[i_domain];
    }
  }
  p_indices[n_domain - 1] = n_documents;
  
  p_head_domains[0] = 0;
  for (int i = 1; i < n_domain; i++) {
    p_head_domains[i] = p_head_domains[i - 1] + p_indices[i - 1];
  }

  p = p_head_domains[n_domain - 1] + n_documents;  // dimension of concated vectors
}

void MCEigendocs::compute()
{
  // Construct count table of words of each documents
  int l = 0;  //todo
  inverse_word_count_table.resize(n_documents);

  {
    Eigen::VectorXi word_count_table(n_documents);
    
    // Initialization
    for (unsigned long long i = 0; i < n_documents; i++) {
      word_count_table(i) = 0;
    }
    
    for (unsigned long long i = 0; i < sentence_lengths[l]; i++) {
      word_count_table(document_id_concated[i]) += 1;
    }
    
    for (unsigned long long i = 0; i < n_documents; i++) {
      inverse_word_count_table[i] = 1.0 / (real)word_count_table(i);
    }
  }

  // Construct matrices: G, H
  VectorXreal G_diag(p);
  G_diag.setZero();
  realSparseMatrix H(p, p);

  construct_matrices(G_diag, H);

  // Construct the matrices for CCA
  std::cout << "Calculate CDMCA..." << std::endl;
  
  realSparseMatrix G_inv_sqrt(p, p);
  construct_h_diag_matrix(G_diag, G_inv_sqrt);

  realSparseMatrix A = (G_inv_sqrt * (H.cast <real> ().cwiseSqrt().selfadjointView<Eigen::Upper>()) * G_inv_sqrt).eval();

  // Execute CDMCA
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<realSparseMatrix> svdA(A, k, 20);
  MatrixXreal principal_components = svdA.matrixV();
  vector_representations = G_inv_sqrt * principal_components.block(0, 0, p, k);
}


void MCEigendocs::construct_matrices (VectorXreal &G_diag, realSparseMatrix &H)
{
  unsigned long long i_sentence_concated = 0;
  unsigned long long n_pushed_triplets = 0;

  unsigned long long n = 0;  // number of data
  for (int i_languages = 0; i_languages < sentence_lengths.size(); i_languages++) {
    n += sentence_lengths[i_languages];
  }
  n += n_documents;

  // todo
  const unsigned long long size_M = sentence_lengths[0];
  VectorXreal M_diag(size_M);
  for (unsigned long long i = 0; i < size_M; i++) {
    if (doc_weighting) {
      M_diag(i) = 1 + inverse_word_count_table[document_id_concated[i]];
    } else {
      M_diag(i) = 2;
    }
    M_diag(i) *= weight_doc_vs_vc;
  }

  std::vector<Triplet> H_tripletList;
  H_tripletList.reserve(TRIPLET_VECTOR_SIZE);

  realSparseMatrix H_temp(p, p);


  // For each languages
  for (int i_languages = 0; i_languages < sentence_lengths.size(); i_languages++) {
      
    const unsigned long long p_v = p_head_domains[2 * i_languages];     // Head of Vi
    const unsigned long long p_c = p_head_domains[2 * i_languages + 1]; // Head of Ci
    const unsigned long long p_d = p_head_domains[n_domain - 1];        // Head of D
    const unsigned long long window_size = window_sizes[i_languages];
    const unsigned long long vocab_size = vocab_sizes[i_languages];
    const unsigned long long sentence_size = sentence_lengths[i_languages];


    // Construct offset table (If window_size=2, offsets = [-2, -1, 1, 2])
    int offsets[2 * window_size];
    fill_offset_table(offsets, window_size);

    // For tokens of each languages
    for (unsigned long long i_sentence = 0; i_sentence < sentence_size; i_sentence++) {

      const unsigned long long word0 = sentence_concated[i_sentence_concated];
      const unsigned long long docid = document_id_concated[i_sentence_concated];
      real H_ij;

      if (doc_weighting) {
        H_ij = inverse_word_count_table[docid];
      } else {
        H_ij = 1;
      }
      H_ij *= weight_doc_vs_vc;

      G_diag(word0 + p_v) += M_diag(i_sentence);
      G_diag(docid + p_d) += 1;

      H_tripletList.push_back(Triplet(word0 + p_v,  docid + p_d,  H_ij));  // Element of t(Wi) %*% Ji

      // For each words of context window
      for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
        const long long i_word1 = i_sentence + offsets[i_offset1];
        const long long i_word1_concated = i_sentence_concated + offsets[i_offset1];
      
        // If `i_word1` is out of indices of sentence
        if ((i_word1 < 0) || (i_word1 >= sentence_size)) continue;
        
        const unsigned long long word1 = sentence_concated[i_word1_concated] + vocab_size * i_offset1;
        
        G_diag(word1 + p_c) += M_diag[i_word1_concated];
        
        H_tripletList.push_back(Triplet(word0 + p_v, word1 + p_c, 1.0));   // Element of t(Wi) %*% Ci
        H_tripletList.push_back(Triplet(word1 + p_c, docid + p_d, H_ij));  // Element of t(Ci) %*% Ji
      }
      
      n_pushed_triplets += 2*window_size + 1;
      
      // Commit temporary matrices
      if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_sentence == sentence_size - 1)) {
        update_crossprod_matrix(H_tripletList, H_temp, H);
        
        n_pushed_triplets = 0;
      }

      i_sentence_concated++;
    }
  }


  H.makeCompressed();

  std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
  std::cout << "H,  " << H.nonZeros() << ",  " << H.rows() << ",  " << H.cols() << std::endl;
  std::cout << std::endl;
}


MatrixXreal MCEigendocs::get_vector_representations() {
  return vector_representations;
}

VectorXreal MCEigendocs::get_singular_values() {
  return singular_values;
}

int MCEigendocs::get_n_domain()
{
  return n_domain;
}

unsigned long long MCEigendocs::get_p()
{
  return p;
}

unsigned long long MCEigendocs::get_p_head_domains(int index)
{
  return p_head_domains[index];
}
