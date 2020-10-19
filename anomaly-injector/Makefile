default:
	g++ -fopenmp -o cpu cpu.cpp
	g++ -fopenmp -o cpu-all-cores cpu-all-cores.cpp
	g++ -fopenmp -o cpu-all-cores-with-affinity cpu-all-cores-with-affinity.cpp
	g++ -fopenmp -o mem mem.c
	g++ -o l3 l3.c -lrt

clean:
	rm src/mem
	rm src/cpu
	rm src/l3
	rm src/cpu-all-cores
	rm src/cpu-each-thread-each-core
