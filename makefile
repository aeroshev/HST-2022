CC=gcc
CXX=g++
CFLAGS=-c -Wall -O2 -Xclang -fopenmp -std=c++17
LDFLAGS=
FOLDER=./server
SOURCES=$(FOLDER)/src
BUILD=$(FOLDER)/build

all: simple

omp_test: $(FOLDER)/hello_world.cpp
	$(CXX) -Xclang -fopenmp $(FOLDER)/hello_world.cpp -lomp -o $(BUILD)/omp.o

simple: main.o matrix.o executor.o
	$(CXX) $(BUILD)/main.o $(BUILD)/matrix.o $(BUILD)/executor.o -o $(BUILD)/hybrid_supercomputer

openmp: main.o matrix.o executor.o
	$(CXX) $(BUILD)/main.o $(BUILD)/matrix.o $(BUILD)/executor.o -Xclang -fopenmp -lomp -o $(BUILD)/hybrid_omp

main.o: $(SOURCES)/main.cpp
	$(CXX) $(CFLAGS) $(SOURCES)/main.cpp -o $(BUILD)/main.o

matrix.o: $(SOURCES)/matrix.cpp
	$(CXX) $(CFLAGS) $(SOURCES)/matrix.cpp -o $(BUILD)/matrix.o

executor.o: $(SOURCES)/executor.cpp
	$(CXX) $(CFLAGS) $(SOURCES)/executor.cpp -o $(BUILD)/executor.o

clean:
	rm -rf $(BUILD)/*.o

run:
	$(BUILD)/hybrid_supercomputer

run_omp:
	$(BUILD)/hybrid_omp

test:
	$(BUILD)/omp.o
