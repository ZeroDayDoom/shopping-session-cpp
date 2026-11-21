CXX ?= g++
CXXFLAGS ?= -O3 -march=x86-64-v3 -std=c++17 -Wall -pipe
LDFLAGS ?= -lpthread

pipeline: src/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f pipeline

.PHONY: clean
