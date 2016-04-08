/*
  Work in Progress
*/

#include "docopt.cpp/docopt.h"
#include "../src/kadingir_core.hpp"
#include "../src/utils.hpp"


static const char USAGE[] =
R"(Kadingir: Cross-Lingual Eigenwords (bilingual)

    Usage:
      kadingir_cleigenwords_bilingual --corpus1 <corpus1> --corpus2 <corpus2> --output <output> --vocab1 <vocab1> --vocab2 <vocab2> --window1 <window1> --window2 <window2> --dim <dim> [--debug]

    Options:
      -h --help            Show this screen.
      --version            Show version.
      --corpus1=<corpus1>  File path of corpus of language 1
      --corpus2=<corpus2>  File path of corpus of language 2
      --output=<output>    File path of output
      --vocab1=<vocab1>    Size of vocabulary of language 1
      --vocab2=<vocab2>    Size of vocabulary of language 2
      --window1=<window1>  Window size of language 1
      --window2=<window2>  Window size of language 2
      --dim=<dim>          Dimension of representation
      --debug              Debug option [default: false]
)";


int main(int argc, const char** argv)
{
  // Parse command line arguments
  std::map<std::string, docopt::value> args
    = docopt::docopt(USAGE, { argv + 1, argv + argc }, true, "Kadingir 1.0");

  const char* path_corpus1 = args["--corpus1"].asString().c_str();
  const char* path_corpus2 = args["--corpus2"].asString().c_str();
  const char* path_output  = args["--output"].asString().c_str();
  const int   n_vocab1     = args["--vocab1"].asLong();
  const int   n_vocab2     = args["--vocab2"].asLong();
  const int   window1      = args["--window1"].asLong();
  const int   window2      = args["--window2"].asLong();
  const int   dim          = args["--dim"].asLong();
  const bool  debug        = args["--debug"].asBool();

  // Build word count table
  MapCounter count_table;
  unsigned long long n_tokens = 0;
  unsigned long long n_documents = 0;

  build_count_table(path_corpus1, count_table, n_documents, n_tokens);
  
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

    if (i_vocab >= n_vocab1) {
      std::cout << "min count:  " << iter_int << ", " << iter_str << std::endl;
      break;
    }
  }

  // Convert words to wordtype id
  unsigned long long n_oov = 0;
  std::vector<int> tokens(n_tokens);

  convert_corpus_to_wordtype(path_corpus1, table_wordtype_id, tokens, n_oov);

  // Display some informations
  std::cout << std::endl;
  std::cout << "Corpus 1    : " << path_corpus1 << std::endl;
  std::cout << "# of tokens : " << n_tokens << std::endl;
  std::cout << "# of OOV    : " << n_oov << std::endl;
  std::cout << "# of vocab  : " << n_vocab1 << std::endl;
  std::cout << "Window size : " << window1 << std::endl;
  std::cout << "Coverage(%) : " << 100 * (n_tokens - n_oov) / (double)n_tokens << std::endl;

  std::cout << "Output      : " << path_output << std::endl;
  std::cout << "dim         : " << dim << std::endl;
  std::cout << std::endl;

  // Execute EigenwordsOSCCA
  EigenwordsOSCCA eigenwords(tokens, window1, n_vocab1, dim, debug);
  eigenwords.compute();
  MatrixXd vectors = eigenwords.get_word_vectors();

  // Output vector representations as a txt file
  std::vector<std::string> wordtypes(n_vocab1);

  for (int i = 0; i < vectors.rows(); i++) {
    if (i == 0) {
      wordtypes[i] = "<OOV>";
    } else {
      wordtypes[i] = count_vector[i - 1].first;
    }
  }

  write_txt(path_output, wordtypes, vectors, n_vocab1, dim);


  return 0;
}
