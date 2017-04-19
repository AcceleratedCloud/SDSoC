all:
	g++ -c LogisticRegression.cpp -D __TEST__
	g++ -c accelerator.cpp
	g++ -Wall -std=c++11 -o LogisticRegression LogisticRegression.o accelerator.o
	rm LogisticRegression.o accelerator.o
	tar --lzma -xvf datasets.tar.lzma

clean:
	> LogisticRegression
	> foo.dat
	> foo.out
	rm LogisticRegression *.dat *.out
