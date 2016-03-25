kadingir: kadingir.cpp kadingir_core.o
	g++ -Wall -I/usr/include/eigen3/ kadingir.cpp -o kadingir kadingir_core.o
kadingir_core.o: ../src/kadingir_core.cpp
	g++ -Wall -I/usr/include/eigen3/ -c ../src/kadingir_core.cpp -o kadingir_core.o
clean:
	rm -f *.o kadingir
