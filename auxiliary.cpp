#include "auxiliary.h"

void set_seed (unsigned int seed)
{
	srand (seed);
}

int rand_int (int from, int to)
{
	return from + rand () % (to + 1);
}

double rand_double (double from, double to)
{
	return from + rand () / (double)(RAND_MAX / to);
}
