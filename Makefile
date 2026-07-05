# ARCH=native en local; en Khipu compilar con `make ARCH=skylake-avx512`
# (nodos standard n003-5 son Skylake; portable y compilable desde el nodo de acceso).
ARCH ?= native

CXX = mpic++
CXXFLAGS = -O3 -Wall -std=c++17 -fopenmp -march=$(ARCH)

CXX_SEQ = g++
CXXFLAGS_SEQ = -O3 -Wall -std=c++17 -march=$(ARCH)

SRC_DIR = src
BIN_DIR = bin

all: directories seq_ranksort seq_quicksort ranksort quicksort hybrid_ranksort hybrid_quicksort

directories:
	mkdir -p $(BIN_DIR)

seq_ranksort: $(SRC_DIR)/seq_ranksort.cpp
	$(CXX_SEQ) $(CXXFLAGS_SEQ) -o $(BIN_DIR)/seq_ranksort $(SRC_DIR)/seq_ranksort.cpp

seq_quicksort: $(SRC_DIR)/seq_quicksort.cpp
	$(CXX_SEQ) $(CXXFLAGS_SEQ) -o $(BIN_DIR)/seq_quicksort $(SRC_DIR)/seq_quicksort.cpp

ranksort: $(SRC_DIR)/mpi_ranksort.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/ranksort $(SRC_DIR)/mpi_ranksort.cpp

quicksort: $(SRC_DIR)/mpi_quicksort.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/quicksort $(SRC_DIR)/mpi_quicksort.cpp

hybrid_ranksort: $(SRC_DIR)/hybrid_ranksort.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/hybrid_ranksort $(SRC_DIR)/hybrid_ranksort.cpp

hybrid_quicksort: $(SRC_DIR)/hybrid_quicksort.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/hybrid_quicksort $(SRC_DIR)/hybrid_quicksort.cpp

clean:
	rm -rf $(BIN_DIR)

