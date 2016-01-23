/* kadingir.cpp */

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>

typedef std::map<std::string, int> MapCounter;
typedef std::pair<std::string, int> PairCounter;
typedef std::vector<PairCounter>::const_iterator PairIterator;

bool sort_greater(const PairCounter& left, const PairCounter& right)
{
  return left.second > right.second;
}


int main()
{
  int n_documents = 0;
  char ch;
  std::string word_temp;
  MapCounter count_table;
  
  std::ifstream fin("../data/text8");
  fin.unsetf(std::ios::skipws);
  
  while (!fin.eof()) {
    fin >> ch;
    if (fin.eof()) break;
    
    if (ch == '\n' || ch == ' ') {
      // If `ch` is a separation of documents or words
      
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
  
  // Sort & display `count_table`.
  std::vector<PairCounter> count_vector(count_table.begin(), count_table.end());  
  std::sort(count_vector.begin(), count_vector.end(), sort_greater);
  
  int n_display = 0;
  for (PairIterator iter = count_vector.begin(); iter != count_vector.end(); iter++) {
    std::cout << iter->second << " " << iter->first << std::endl;
    n_display++;
    if (n_display > 100) break;
  }
  
  return 0;
}
