all:
	g++ -c KMeans.cpp -D __TEST__
	g++ -c accelerator.cpp -O3
	g++ -Wall -std=c++11 -o KMeans KMeans.o accelerator.o
	rm KMeans.o accelerator.o

clean:
	rm -f KMeans *.out
