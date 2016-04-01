/* kadingir.cpp */

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include "docopt.cpp/docopt.h"
#include "../src/kadingir_core.hpp"


typedef std::map<std::string, int> MapCounter;
typedef std::pair<std::string, int> PairCounter;


static const char USAGE[] =
R"(Kadingir: Eigenwords (OSCCA)

    Usage:
      kadingir --corpus <corpus> --output <output> --vocab <vocab> --dim <dim> --window <window> [--debug]

    Options:
      -h --help          Show this screen.
      --version          Show version.
      --corpus=<corpus>  File path of corpus
      --output=<output>  File path of output
      --vocab=<vocab>    Size of vocabulary
      --dim=<dim>        Dimension of representation
      --window=<window>  Window size
      --debug            Debug option [default: false]
)";

bool sort_greater(const PairCounter& left, const PairCounter& right)
{
  return left.second > right.second;
}

void build_count_table(const char* path_corpus, MapCounter &count_table,
                       unsigned long long &n_documents, unsigned long long &n_tokens)
{
  std::ifstream fin;
  fin.unsetf(std::ios::skipws);
  fin.open(path_corpus);

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

void convert_corpus_to_wordtype(const char* path_corpus, MapCounter &table_wordtype_id,
                                std::vector<int> &tokens, unsigned long long &n_oov)
{
  std::ifstream fin;
  fin.unsetf(std::ios::skipws);
  fin.open(path_corpus);

  char ch;
  std::string word_temp;
  unsigned long long i_tokens = 0;

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

        i_tokens += 1;
        word_temp.erase();
      }
    } else {
      // If `ch` is a character of a word
      word_temp += ch;
    }  
  }
  fin.close();
}

void write_txt(const char* path_output, const std::vector<std::string> &wordtypes,
               MatrixXd &vectors,
               const unsigned long long n_vocab, const int dim)
{
  std::ofstream file_output;
  file_output.open(path_output, std::ios::out);
  file_output << n_vocab << " " << dim << std::endl;

  for (int i = 0; i < vectors.rows(); i++) {
    file_output << wordtypes[i] << " ";
    for (int j = 0; j < vectors.cols(); j++) {
      file_output << vectors(i, j) << " ";
    }
    file_output << std::endl;
  }
}

int main(int argc, const char** argv)
{
  // Parse command line arguments
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, { argv + 1, argv + argc }, true, "Kadingir 1.0");

  const char* path_corpus = args["--corpus"].asString().c_str();
  const char* path_output = args["--output"].asString().c_str();
  const int   n_vocab     = args["--vocab"].asLong();
  const int   dim         = args["--dim"].asLong();
  const int   window      = args["--window"].asLong();
  const bool  debug       = args["--debug"].asBool();

  // Build word count table
  MapCounter count_table;
  unsigned long long n_tokens = 0;
  unsigned long long n_documents = 0;

  build_count_table(path_corpus, count_table, n_documents, n_tokens);
  
  // Sort `count_table`.
  std::vector<PairCounter> count_vector(count_table.begin(), count_table.end());
  std::sort(count_vector.begin(), count_vector.end(), sort_greater);
  
  // Construct table (std::string)word -> (int)wordtype id
  unsigned long long i_vocab = 1;
  MapCounter table_wordtype_id;
  for (auto iter = count_vector.begin(); iter != count_vector.end(); iter++) {
    std::string iter_str = iter->first;
    int iter_int = iter->second;
    table_wordtype_id.insert(PairCounter(iter_str, i_vocab));
    i_vocab++;

    if (i_vocab >= n_vocab) {
      std::cout << "min count:  " << iter_int << ", " << iter_str << std::endl;
      break;
    }
  }

  // Convert words to wordtype id
  unsigned long long n_oov = 0;
  std::vector<int> tokens(n_tokens);

  convert_corpus_to_wordtype(path_corpus, table_wordtype_id, tokens, n_oov);

  // Display some informations
  std::cout << std::endl;
  std::cout << "Corpus      : " << path_corpus << std::endl;
  std::cout << "Output      : " << path_output << std::endl;
  std::cout << "# of tokens : " << n_tokens << std::endl;
  std::cout << "# of OOV    : " << n_oov << std::endl;
  std::cout << "# of vocab  : " << n_vocab << std::endl;
  std::cout << "Coverage(%) : " << 100 * (n_tokens - n_oov) / (double)n_tokens << std::endl;
  std::cout << "dim         : " << dim << std::endl;
  std::cout << "Window size : " << window << std::endl;
  std::cout << std::endl;

  // Execute EigenwordsOSCCA
  EigenwordsOSCCA eigenwords(tokens, window, n_vocab, dim, debug);
  eigenwords.compute();
  MatrixXd vectors = eigenwords.get_word_vectors();

  // Output vector representations as a txt file
  std::vector<std::string> wordtypes(n_vocab);

  for (int i = 0; i < vectors.rows(); i++) {
    if (i == 0) {
      wordtypes[i] = "<OOV>";
    } else {
      wordtypes[i] = count_vector[i - 1].first;
    }
  }

  write_txt(path_output, wordtypes, vectors, n_vocab, dim);


  return 0;
}
