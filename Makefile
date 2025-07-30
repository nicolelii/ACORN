CXX = g++
CXXFLAGS = -std=c++11 -g -O3 -I. -I./build -fopenmp
LDFLAGS = -L./build/faiss -lfaiss
SRC = acorn_batching_sweep.cpp
TARGET = acorn_batch.bin
SIFT_DIR = sift

$(TARGET): $(SRC) | sift
	$(CXX) $(CXXFLAGS) $(SRC) $(LDFLAGS) -o $(TARGET)

sift:
	@if [ ! -d "$(SIFT_DIR)" ]; then \
		echo "Running setup script..."; \
		scripts/download_sift.sh; \
	fi

clean:
	rm -f $(TARGET)

