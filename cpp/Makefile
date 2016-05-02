CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3
INCLUDES = -I"/usr/include/"

all: kadingir_oscca kadingir_cleigenwords_bilingual
kadingir_oscca: kadingir_oscca.cpp kadingir_core.o docopt.o utils.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) utils.o docopt.o kadingir_oscca.cpp -o kadingir_oscca kadingir_core.o
kadingir_cleigenwords_bilingual: kadingir_cleigenwords_bilingual.cpp kadingir_core.o docopt.o utils.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) utils.o docopt.o kadingir_cleigenwords_bilingual.cpp -o kadingir_cleigenwords_bilingual kadingir_core.o
utils.o: ../src/utils.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c ../src/utils.cpp -o utils.o
kadingir_core.o: ../src/kadingir_core.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c ../src/kadingir_core.cpp -o kadingir_core.o
docopt.o: docopt.cpp/docopt.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c docopt.cpp/docopt.cpp -o docopt.o
clean:
	rm -f *.o kadingir_oscca
