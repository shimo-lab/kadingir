#ifndef __KADINGIR_UTILS_HPP__
#define __KADINGIR_UTILS_HPP__


#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <string>

typedef std::map<std::string, int> MapCounter;
typedef std::pair<std::string, int> PairCounter;


bool sort_greater(const PairCounter& left, const PairCounter& right);
void build_count_table(const std::string &path_corpus, MapCounter &count_table,
                       unsigned long long &n_documents, unsigned long long &n_tokens);
void convert_corpus_to_wordtype(const std::string &path_corpus, MapCounter &table_wordtype_id,
                                std::vector<int> &tokens, std::vector<int> &document_id,
				unsigned long long &n_oov);
std::string replace_char(std::string str, const char ch1, const char ch2);
void write_txt(const std::string &path_output,
               const std::vector<std::string> &wordtypes,
               const Eigen::MatrixXd &vectors,
               const unsigned long long n_vocab, const int dim);


#endif
