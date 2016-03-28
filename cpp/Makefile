CXX = g++-4.9
CXXFLAGS = --std=c++11 -Wall
INCLUDES = -I/usr/include/eigen3/

kadingir_oscca: kadingir_oscca.cpp kadingir_core.o docopt.cpp/docopt.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) docopt.cpp/docopt.cpp kadingir_oscca.cpp -o kadingir_oscca kadingir_core.o
kadingir_core.o: ../src/kadingir_core.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c ../src/kadingir_core.cpp -o kadingir_core.o
clean:
	rm -f *.o kadingir_oscca
