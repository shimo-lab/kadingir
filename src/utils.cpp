
#include "utils.hpp"


bool sort_greater(const PairCounter& left, const PairCounter& right)
{
  return left.second > right.second;
}

void build_count_table(const std::string &path_corpus, MapCounter &count_table,
                       unsigned long long &n_documents, unsigned long long &n_tokens)
{
  std::ifstream fin;
  fin.unsetf(std::ios::skipws);
  fin.open(path_corpus.c_str());

  char ch;
  std::string word_temp;
  
  while (!fin.eof()) {
    fin >> ch;
    if (fin.eof()) break;
    
    if (ch == '\n' || ch == ' ') {
      // If `ch` is a separation of documents or words
      
      n_tokens += 1;

      if (ch == '\n') {
        n_documents += 1;
      }
      
      if (!word_temp.empty()) {
        if (count_table.count(word_temp) == 0) {
          count_table.insert(PairCounter(word_temp, 1));
        } else {
          count_table[word_temp] += 1;
        }
        word_temp.erase();
      }
      
    } else {
      // If `ch` is a character of a word
      word_temp += ch;
    }
  }
  fin.close();
}

void convert_corpus_to_wordtype(const std::string &path_corpus, MapCounter &table_wordtype_id,
                                std::vector<int> &tokens, std::vector<int> &document_id,
				unsigned long long &n_oov)
{
  std::ifstream fin;
  fin.unsetf(std::ios::skipws);
  fin.open(path_corpus.c_str());

  char ch;
  std::string word_temp;
  unsigned long long i_tokens = 0, i_documents = 0;

  while (!fin.eof()) {
    fin >> ch;
    if (fin.eof()) break;

    if (ch == '\n' || ch == ' ') {
      // If `ch` is a separation of documents or words
      if (!word_temp.empty()) {
        if (table_wordtype_id.count(word_temp) == 0) {
          // If the token is Out of Vocabulary
          tokens[i_tokens] = 0;
          n_oov += 1;
        } else {
          // Otherwise
          tokens[i_tokens] = table_wordtype_id[word_temp];
        }
	document_id[i_tokens] = i_documents;

        i_tokens += 1;
        word_temp.erase();
      } else {
	if (ch == '\n') {
	  i_documents += 1;
	}
      }
    } else {
      // If `ch` is a character of a word
      word_temp += ch;
    }  
  }
  fin.close();
}

void write_txt(const std::string &path_output, const std::vector<std::string> &wordtypes,
               Eigen::MatrixXd &vectors,
               const unsigned long long n_vocab, const int dim)
{
  std::ofstream file_output;
  file_output.open(path_output.c_str(), std::ios::out);
  file_output << n_vocab << " " << dim << std::endl;

  for (int i = 0; i < vectors.rows(); i++) {
    file_output << wordtypes[i] << " ";
    for (int j = 0; j < vectors.cols(); j++) {
      file_output << vectors(i, j) << " ";
    }
    file_output << std::endl;
  }
}
