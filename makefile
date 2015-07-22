CC=g++
CFLAGS=-c -std=c++11 -O3 -Wall -Wno-sign-compare 
LDFLAGS=
SOURCES = NeuralNet.cpp\
	auxiliary.cpp\
	main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=main

all:clean0 $(SOURCES) $(EXECUTABLE) clean1
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -fopenmp -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -fopenmp -o $@

clean0:
	rm -rf *.o main
clean1:
	rm -rf *.o
