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
	//std::cout << index1 << " " << index2 << std::endl; 
	A tmp = a [index1];
	a [index1] = a [index2];
	a [index2] = tmp;
}

template <typename T>
void shuffle (std::vector<T>& a)
{
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	set_seed (std::chrono::duration_cast<std::chrono::nanoseconds> (end - start).count ());
	int count = 0;
	for (int i = 0; i < a.size () - 1; ++i) {
		//printf ("%d ", rand_int (0, 10));
		swap (a, rand_int (0, a.size () - count - 2), a.size () - count -1);
		count++;
	}
}



