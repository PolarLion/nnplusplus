#include <vector>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <ratio>



int rand_int (int from, int to);

double rand_double (double from, double to);

void set_seed (unsigned int seed);

template <typename A>
void swap (std::vector<A>& a, int index1, int index2)
{ 
	std::cout << index1 << " " << index2 << std::endl; 
	A tmp = a [index1];
	a [index1] = a [index2];
	a [index2] = tmp;
}

template <typename T>
void shuffle (std::vector<T>& a)
{
	//set_seed (time (NULL));
	//std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	//std::chrono::duration<unsigned int> now = start;
	//std::chrono::system_clock::time
	//set_seed (start.count ());
	int count = 0;
	for (int i = 0; i < a.size () - 1; ++i) {
		//printf ("%d ", rand_int (0, 10));
		swap (a, rand_int (0, a.size () - count - 2), a.size () - count -1);
		count++;
	}
}



