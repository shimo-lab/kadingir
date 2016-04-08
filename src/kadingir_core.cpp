/*
 * kadingir_core.cpp
 *
 * memo :
 *  - `0` in id_wordtype indicates `<OOV>` (Out of Vocabulary, tokens that are not included in vocabulary).
 *  - tWC indicates matrix multiplication (crossprod) of W and C (In R, tWC = t(W) %*% C).
 *  - `_h` in `tWW_h` means "cast, diagonal, cwiseInverse, cwizeSqrt, cwiseSqrt"
 */


#include "kadingir_core.hpp"

const int TRIPLET_VECTOR_SIZE = 10000000;


// Update crossprod matrix using triplets
template <class MatrixX>
void update_crossprod_matrix(std::vector<Triplet> &X_tripletList, MatrixX &X_temp, MatrixX &X)
{
  X_temp.setFromTriplets(X_tripletList.begin(), X_tripletList.end());
  X_tripletList.clear();
  X += X_temp;
  X_temp.setZero();
}

// If window_size=2, it returns [-2, -1, 1, 2].
std::vector<int> generate_offset_table(int window_size)
{
  std::vector<int> offsets(2 * window_size);

  int i_offset1 = 0;
  for (int offset = -window_size; offset <= window_size; offset++){
    if (offset != 0) {
      offsets[i_offset1] = offset;
      i_offset1++;
    }
  }

  return offsets;
}

dSparseMatrix asDiagonalDSparseMatrix(const VectorXd &x)
{
  int n = x.size();
  dSparseMatrix x_diag(n, n);

  for (int i = 0; i < n; i++) {
    x_diag.insert(i, i) = x(i);
  }

  return x_diag;
}


EigenwordsOSCCA::EigenwordsOSCCA(
  const std::vector<int>& _id_wordtype,
  const int _window_size,
  const int _vocab_size,
  const int _k,
  const bool _debug
) : id_wordtype(_id_wordtype),
    window_size(_window_size),
    vocab_size(_vocab_size),
    k(_k),
    debug(_debug)
{
  c_col_size = 2 * (unsigned long long)window_size * vocab_size;

  tWW_diag.resize(vocab_size);
  tCC_diag.resize(c_col_size);
  tWC.resize(vocab_size, c_col_size);
}

void EigenwordsOSCCA::compute()
{
  construct_matrices();
  dSparseMatrix tWW_h_diag = asDiagonalDSparseMatrix(tWW_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  dSparseMatrix tCC_h_diag = asDiagonalDSparseMatrix(tCC_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());

  std::cout << "Calculate OSCCA..." << std::endl;
  dSparseMatrix a = tWW_h_diag * (tWC.cast <double> ().eval().cwiseSqrt()) * tCC_h_diag;
  
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdA(a, k, 20);
  
  word_vectors    = tWW_h_diag * svdA.matrixU();
  context_vectors = tCC_h_diag * svdA.matrixV();
  singular_values = svdA.singularValues();
  
  if (!debug) {
    tWW_diag.resize(0);
    tCC_diag.resize(0);
    tWC.resize(0, 0);
  }
}

void EigenwordsOSCCA::construct_matrices()
{
  const unsigned long long id_wordtype_size = id_wordtype.size();
  unsigned long long n_pushed_triplets = 0;
  std::vector<int> offsets = generate_offset_table(window_size);
  
  std::vector<Triplet> tWC_tripletList(TRIPLET_VECTOR_SIZE);
  iSparseMatrix tWC_temp(vocab_size, c_col_size);

  tWW_diag.setZero();
  tCC_diag.setZero();  
  
  for (unsigned long long i_id_wordtype = 0; i_id_wordtype < id_wordtype_size; i_id_wordtype++) {
    const unsigned long long word0 = id_wordtype[i_id_wordtype];
    tWW_diag(word0) += 1;
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      const long long i_word1 = i_id_wordtype + offsets[i_offset1];
      
      // If `i_word1` is out of indices of id_wordtype
      if ((i_word1 < 0) || (i_word1 >= id_wordtype_size)) continue;
      
      const unsigned long long word1 = id_wordtype[i_word1] + vocab_size * i_offset1;

      tCC_diag(word1) += 1;
      tWC_tripletList.push_back(Triplet(word0, word1, 1));
    }
    
    n_pushed_triplets++;
    
    // Commit temporary matrices
    if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_id_wordtype == id_wordtype_size - 1)) {
      update_crossprod_matrix(tWC_tripletList, tWC_temp, tWC);
      n_pushed_triplets = 0;
    }
  }
  
  tWC.makeCompressed();

  std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
  std::cout << "tWC,  " << tWC.nonZeros() << ",  " << tWC.rows() << ",  " << tWC.cols() << std::endl;
  std::cout << std::endl;
}


EigenwordsTSCCA::EigenwordsTSCCA(
  const std::vector<int>& _id_wordtype,
  const int _window_size,
  const int _vocab_size,
  const int _k,
  const bool _debug
  ) : id_wordtype(_id_wordtype),
      window_size(_window_size),
      vocab_size(_vocab_size),
      k(_k),
      debug(_debug)
{
  lr_col_size = (unsigned long long)window_size * vocab_size;
  c_col_size = 2 * lr_col_size;
}

void EigenwordsTSCCA::compute()
{
  // Construct crossprod matrices
  tWW_diag.resize(vocab_size);
  tWC.resize(vocab_size, c_col_size);
  tLL.resize(lr_col_size, lr_col_size);
  tLR.resize(lr_col_size, lr_col_size);
  tRR.resize(lr_col_size, lr_col_size);

  construct_matrices();
  run_tscca();
}

void EigenwordsTSCCA::construct_matrices()
{
  const unsigned long long id_wordtype_size = id_wordtype.size();
  unsigned long long n_pushed_triplets = 0;
  std::vector<int> offsets = generate_offset_table(window_size);

  std::vector<Triplet> tWC_tripletList(TRIPLET_VECTOR_SIZE);
  std::vector<Triplet> tLL_tripletList(TRIPLET_VECTOR_SIZE);
  std::vector<Triplet> tLR_tripletList(TRIPLET_VECTOR_SIZE);
  std::vector<Triplet> tRR_tripletList(TRIPLET_VECTOR_SIZE);
  iSparseMatrix tWC_temp(vocab_size, c_col_size);
  iSparseMatrix tLL_temp(lr_col_size, lr_col_size);
  iSparseMatrix tLR_temp(lr_col_size, lr_col_size);
  iSparseMatrix tRR_temp(lr_col_size, lr_col_size);
  tWW_diag.setZero();

  for (unsigned long long i_id_wordtype = 0; i_id_wordtype < id_wordtype_size; i_id_wordtype++) {
    const unsigned long long word0 = id_wordtype[i_id_wordtype];
    tWW_diag(word0) += 1;
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      const long long i_word1 = i_id_wordtype + offsets[i_offset1];
      
      // If `i_word1` is out of indices of id_wordtype
      if ((i_word1 < 0) || (i_word1 >= id_wordtype_size)) continue;
      
      const unsigned long long word1 = id_wordtype[i_word1] + vocab_size * i_offset1;
      
      // Two step CCA
      for (int i_offset2 = 0; i_offset2 < 2 * window_size; i_offset2++) {
        const long long i_word2 = i_id_wordtype + offsets[i_offset2];
        
        // If `i_word2` is out of indices of id_wordtype
        if ((i_word2 < 0) || (i_word2 >= id_wordtype_size)) continue;
        
        const unsigned long long word2 = id_wordtype[i_word2] + vocab_size * i_offset2;
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
      
      tWC_tripletList.push_back(Triplet(word0, word1, 1));
    }
    
    n_pushed_triplets++;
    
    // Commit temporary matrices
    if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_id_wordtype == id_wordtype_size - 1)) {
      update_crossprod_matrix(tWC_tripletList, tWC_temp, tWC);
      update_crossprod_matrix(tLL_tripletList, tLL_temp, tLL);
      update_crossprod_matrix(tLR_tripletList, tLR_temp, tLR);
      update_crossprod_matrix(tRR_tripletList, tRR_temp, tRR);
      
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

// Execute Two Step CCA
void EigenwordsTSCCA::run_tscca()
{
  std::cout << "Calculate TSCCA..." << std::endl;
  
  // Two Step CCA : Step 1
  VectorXi tLL_diag = tLL.diagonal();
  VectorXi tRR_diag = tRR.diagonal();
  dSparseMatrix tLL_h_diag = asDiagonalDSparseMatrix(tLL_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  dSparseMatrix tRR_h_diag = asDiagonalDSparseMatrix(tRR_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  dSparseMatrix b = tLL_h_diag * (tLR.cast <double> ().eval().cwiseSqrt()) * tRR_h_diag;
  
  std::cout << "# of nonzero,  # of rows,  # of cols = " << b.nonZeros() << ",  " << b.rows() << ",  " << b.cols() << std::endl;
  
  std::cout << "Calculate Randomized SVD (1/2)..." << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdB(b, k, 20);
  b.resize(0, 0);  // Release memory
  
  MatrixXd phi_l = svdB.matrixU();
  MatrixXd phi_r = svdB.matrixV();
  
  // Two Step CCA : Step 2
  VectorXd tSS_h1 = (phi_l.transpose() * (tLL.cast <double> ().selfadjointView<Eigen::Upper>() * phi_l)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt();
  VectorXd tSS_h2 = (phi_r.transpose() * (tRR.cast <double> ().selfadjointView<Eigen::Upper>() * phi_r)).eval().diagonal().cwiseInverse().cwiseSqrt().cwiseSqrt();
  
  // Release memory
  tLL.resize(0, 0);
  tRR.resize(0, 0);
  
  VectorXd tSS_h(2*k);
  tSS_h << tSS_h1, tSS_h2;
  dSparseMatrix tSS_h_diag = asDiagonalDSparseMatrix(tSS_h);
  MatrixXd tWS(vocab_size, 2*k);
  tWS << tWC.topLeftCorner(vocab_size, lr_col_size).cast <double> ().cwiseSqrt() * phi_l, tWC.topRightCorner(vocab_size, lr_col_size).cast <double> ().cwiseSqrt() * phi_r;
  tWW_h_diag = asDiagonalDSparseMatrix(tWW_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  MatrixXd a = tWW_h_diag * tWS * tSS_h_diag;
  
  std::cout << "Calculate Randomized SVD (2/2)..." << std::endl;
  RedSVD::RedSVD<MatrixXd> svdA(a, k, 20);

  word_vectors = tWW_h_diag * svdA.matrixU();
  context_vectors = tSS_h_diag * svdA.matrixV();
  singular_values = svdA.singularValues();
}


Eigendocs::Eigendocs(
    const std::vector<int>& _id_wordtype,
    const std::vector<int>& _id_document,
    const int _window_size,
    const int _vocab_size,
    const int _k,
    const bool _link_w_d,
    const bool _link_c_d,
    const bool _debug
  ) : id_wordtype(_id_wordtype),
      id_document(_id_document),
      window_size(_window_size),
      vocab_size(_vocab_size),
      k(_k),
      link_w_d(_link_w_d),
      link_c_d(_link_c_d),
      debug(_debug)
{
  lr_col_size = (unsigned long long)window_size * vocab_size;
  c_col_size = 2 * lr_col_size;

  p_head_domains[0] = 0;
  p_head_domains[1] = vocab_size;
  p_head_domains[2] = vocab_size + c_col_size;
}

void Eigendocs::compute()
{
  const unsigned long long n_documents = *std::max_element(id_document.begin(), id_document.end()) + 1;

  p_indices[0] = vocab_size;
  p_indices[1] = c_col_size;
  p_indices[2] = n_documents;
  p = vocab_size + c_col_size + n_documents;

  // Construct crossprod matrices
  tWW_diag.resize(vocab_size);
  tCC_diag.resize(c_col_size);
  tDD_diag.resize(n_documents);
  H.resize(p, p);
  construct_matrices();

  // Construct the matrices for CCA and execute CCA  
  G_diag.resize(p);
  
  if (link_w_d && link_c_d) {
    G_diag << 2*tWW_diag, 2*tCC_diag, 2*tDD_diag;
  } else if (!link_w_d) {
    G_diag <<   tWW_diag, 2*tCC_diag,   tDD_diag;
  } else {
    G_diag << 2*tWW_diag,   tCC_diag,   tDD_diag;
  }

  dSparseMatrix G_inv_sqrt = asDiagonalDSparseMatrix(G_diag.cast <double> ().cwiseInverse().cwiseSqrt().cwiseSqrt());
  G_inv_sqrt /= sqrt(2);

  dSparseMatrix A = (G_inv_sqrt * ((H.cast <double> ().cwiseSqrt()).selfadjointView<Eigen::Upper>()) * G_inv_sqrt).eval();

  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdA(A, k, 20);
  MatrixXd principal_components = svdA.matrixV();
  vector_representations = G_inv_sqrt * principal_components.block(0, 0, p, k);
  eigenvalues = svdA.singularValues();  // singular values of symmetric matrix A is the same as its eigenvalues.
  
  if (!debug) {
      G_diag.resize(0);
      H.resize(0, 0);
  }
}

void Eigendocs::construct_matrices()
{
  const unsigned long long id_wordtype_size = id_wordtype.size();
  unsigned long long n_pushed_triplets = 0;
  std::vector<int> offsets = generate_offset_table(window_size);

  std::vector<Triplet> H_tripletList(TRIPLET_VECTOR_SIZE);
  iSparseMatrix H_temp(p, p);

  tWW_diag.setZero();
  tCC_diag.setZero();
  tDD_diag.setZero();
    
  for (unsigned long long i_id_wordtype = 0; i_id_wordtype < id_wordtype_size; i_id_wordtype++) {
    const unsigned long long word0 = id_wordtype[i_id_wordtype];
    const unsigned long long d_id  = id_document[i_id_wordtype];

    tWW_diag(word0) += 1;
    tDD_diag(d_id) += 1;
    
    if (link_w_d) {
      H_tripletList.push_back(Triplet(word0,  p_head_domains[2] + d_id,  1));  // Element of tWD
    }
    
    for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
      const long long i_word1 = i_id_wordtype + offsets[i_offset1];
      
      // If `i_word1` is out of indices of id_wordtype
      if ((i_word1 < 0) || (i_word1 >= id_wordtype_size)) continue;
      
      const unsigned long long word1 = id_wordtype[i_word1] + vocab_size * i_offset1;

      tCC_diag(word1) += 1;

      H_tripletList.push_back(Triplet(word0, p_head_domains[1] + word1, 1));  // Element of tWC
      if (link_c_d) {
        H_tripletList.push_back(Triplet(p_head_domains[1] + word1, p_head_domains[2] + d_id, 1));  // Element of tCD
      }
    }
    
    n_pushed_triplets += 2*window_size + 1;
    
    // Commit temporary matrices
    if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_id_wordtype == id_wordtype_size - 1)) {
      update_crossprod_matrix(H_tripletList, H_temp, H);
      n_pushed_triplets = 0;
    }
  }

  H.makeCompressed();

  std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
  std::cout << "H,  " << H.nonZeros() << ",  " << H.rows() << ",  " << H.cols() << std::endl;
  std::cout << std::endl;
}



CLEigenwords::CLEigenwords(
  const std::vector<int>& _id_wordtype_concated,
  const std::vector<int>& _id_document_concated,
  const std::vector<int> _window_sizes,
  const std::vector<int> _vocab_sizes,
  const std::vector<unsigned long long> _id_wordtype_lengths,
  const int _k,
  const bool _link_v_c,
  const bool _weighting_tf,
  const std::vector<double> _weight_vsdoc,
  const bool _debug
  ) : id_wordtype_concated(_id_wordtype_concated),
      id_document_concated(_id_document_concated),
      window_sizes(_window_sizes),
      vocab_sizes(_vocab_sizes),
      id_wordtype_lengths(_id_wordtype_lengths),
      k(_k),
      link_v_c(_link_v_c),
      weighting_tf(_weighting_tf),
      weight_vsdoc(_weight_vsdoc),
      debug(_debug)
{
  if (window_sizes.size() != vocab_sizes.size()) {
    std::cout << "window_sizes.size() != vocab_sizes.size()" << std::endl;
  }
  n_languages = window_sizes.size();
  
  // dimension of context domain
  lr_col_sizes.resize(n_languages);
  c_col_sizes.resize(n_languages);
  for (int i = 0; i < n_languages; i++) {
    lr_col_sizes[i] = (unsigned long long)window_sizes[i] * vocab_sizes[i];
    c_col_sizes[i] = 2 * lr_col_sizes[i];
  }

  n_documents = *std::max_element(id_document_concated.begin(), id_document_concated.end()) + 1;
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

void CLEigenwords::compute()
{
  if (weighting_tf) {
    // Reweight matching weights using Term-Frequency
    construct_inverse_word_count_table();
  }

  // Construct matrices: G, H
  G_diag.resize(p);
  G_diag.setZero();
  H.resize(p, p);
  construct_matrices();

  std::cout << "Calculate CDMCA..." << std::endl;
  
  dSparseMatrix G_inv_sqrt = asDiagonalDSparseMatrix(G_diag.cwiseInverse().cwiseSqrt().cwiseSqrt());
  dSparseMatrix A = (G_inv_sqrt * (H.cast <double> ().cwiseSqrt().selfadjointView<Eigen::Upper>()) * G_inv_sqrt).eval();

  if (!debug) {
    G_diag.resize(0);
    H.resize(0, 0);
  }

  // Execute CDMCA
  std::cout << "Calculate Randomized SVD..." << std::endl;
  RedSVD::RedSVD<dSparseMatrix> svdA(A, k, 20);
  MatrixXd principal_components = svdA.matrixV();
  vector_representations = G_inv_sqrt * principal_components.block(0, 0, p, k);
  eigenvalues = svdA.singularValues();  // singular values of symmetric matrix A is the same as its eigenvalues.
}


void CLEigenwords::construct_inverse_word_count_table()
{
  // Construct count table of words of each documents
  VectorXi word_count_table(n_documents);
  inverse_word_count_table.resize(n_languages);
  unsigned long long sum_id_wordtype_lengths = 0;
  
  for (int i_languages = 0; i_languages < n_languages; i_languages++) {
    inverse_word_count_table[i_languages].resize(n_documents);
    
    for (unsigned long long i = 0; i < n_documents; i++) {
      // Initialization
      word_count_table(i) = 0;
    }
    
    for (unsigned long long i = 0; i < id_wordtype_lengths[i_languages]; i++) {
      word_count_table(id_document_concated[sum_id_wordtype_lengths + i]) += 1;
    }
    sum_id_wordtype_lengths += id_wordtype_lengths[i_languages];
    
    for (unsigned long long i = 0; i < n_documents; i++) {
      inverse_word_count_table[i_languages][i] = 1.0 / (double)word_count_table(i);
    }
  }
}


void CLEigenwords::construct_matrices()
{
  unsigned long long sum_id_wordtype_lengths = 0;
  const int weight_v_c = (int)link_v_c;
  
  // Calculate diagonal elements of M  
  std::vector<std::vector<double> > m_diag_languages(n_languages);

  for (int i_languages = 0; i_languages < n_languages; i_languages++) {
    const unsigned long long id_wordtype_length = id_wordtype_lengths[i_languages];
    m_diag_languages[i_languages].resize(id_wordtype_length);
    
    for (unsigned long long i = 0; i < id_wordtype_length; i++) {
      const int id_document = id_document_concated[sum_id_wordtype_lengths + i];
      
      if (id_document >= 0) {
        // From bilingual corpus
        if (weighting_tf) {
          m_diag_languages[i_languages][i] = weight_v_c + weight_vsdoc[i_languages] * inverse_word_count_table[i_languages][id_document];
        } else {
          m_diag_languages[i_languages][i] = weight_v_c + weight_vsdoc[i_languages];
        }
      } else {
        // From monolingual corpus
        m_diag_languages[i_languages][i] = weight_v_c;
      }
    }
    
    sum_id_wordtype_lengths += id_wordtype_length;
  }

  std::vector<Triplet> H_tripletList(TRIPLET_VECTOR_SIZE);
  dSparseMatrix H_temp(p, p);

  unsigned long long i_id_wordtype_concated = 0;
  unsigned long long n_pushed_triplets = 0;
  
  // For each languages
  for (int i_languages = 0; i_languages < n_languages; i_languages++) {
    const unsigned long long p_v = p_head_domains[2 * i_languages];     // Head of Vi
    const unsigned long long p_c = p_head_domains[2 * i_languages + 1]; // Head of Ci
    const unsigned long long p_d = p_head_domains[n_domain - 1];        // Head of D
    const unsigned long long window_size = window_sizes[i_languages];
    const unsigned long long vocab_size = vocab_sizes[i_languages];
    const unsigned long long id_wordtype_size = id_wordtype_lengths[i_languages];
    std::vector<int> offsets = generate_offset_table(window_size);

    // For all tokens of a certain language
    // In following comments, `l` indicates index of languages (i.e. `i_languages`).
    for (unsigned long long i_id_wordtype = 0; i_id_wordtype < id_wordtype_size; i_id_wordtype++) {

      const int word0 = id_wordtype_concated[i_id_wordtype_concated];
      const int docid = id_document_concated[i_id_wordtype_concated];
      double H_ij_vsdoc;  // J^{(l)}_{i_id_wordtype, docid}

      if (docid >= 0) {
        if (weighting_tf) {
          H_ij_vsdoc = weight_vsdoc[i_languages] * inverse_word_count_table[i_languages][docid];
        } else {
          H_ij_vsdoc = weight_vsdoc[i_languages];
        }
      } else {
        H_ij_vsdoc = 0;
      }

      G_diag(word0 + p_v) += m_diag_languages[i_languages][i_id_wordtype];  // Element of t(W_l) %*% W_l
      G_diag(docid + p_d) += 2 * H_ij_vsdoc;  // Element of t(D) %*% D

      H_tripletList.push_back(Triplet(word0 + p_v,  docid + p_d,  H_ij_vsdoc));  // Element of t(W_l) %*% J_l

      // For each words of context window
      for (int i_offset1 = 0; i_offset1 < 2 * window_size; i_offset1++) {
        const long long i_word1 = i_id_wordtype + offsets[i_offset1];
        const long long i_word1_concated = i_id_wordtype_concated + offsets[i_offset1];
      
        // If `i_word1` is out of indices of id_wordtype
        if ((i_word1 < 0) || (i_word1 >= id_wordtype_size)) continue;
        
        const unsigned long long word1 = id_wordtype_concated[i_word1_concated] + vocab_size * i_offset1;
        
        G_diag(word1 + p_c) += m_diag_languages[i_languages][i_id_wordtype];  // Element of t(C_l) %*% C_l
        
        H_tripletList.push_back(Triplet(word0 + p_v, word1 + p_c, weight_v_c));  // Element of t(W_l) %*% C_l
        H_tripletList.push_back(Triplet(word1 + p_c, docid + p_d, H_ij_vsdoc));  // Element of t(C_l) %*% J_l
      }
      
      n_pushed_triplets += 2*window_size + 1;
      
      // Commit temporary matrices
      if ((n_pushed_triplets >= TRIPLET_VECTOR_SIZE - 3*window_size) || (i_id_wordtype == id_wordtype_size - 1)) {
        update_crossprod_matrix(H_tripletList, H_temp, H);
	n_pushed_triplets = 0;
      }

      i_id_wordtype_concated++;
    }
  }

  H.makeCompressed();

  std::cout << "matrix,  # of nonzero,  # of rows,  # of cols" << std::endl;
  std::cout << "H,  " << H.nonZeros() << ",  " << H.rows() << ",  " << H.cols() << std::endl;
  std::cout << std::endl;
}
