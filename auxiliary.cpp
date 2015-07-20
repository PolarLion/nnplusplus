#include "auxiliary.h"


void nnplusplus::set_seed (unsigned int seed)
{
  srand (seed);
}

int nnplusplus::rand_int (int from, int to)
{
  return from + rand () % (to + 1);
}

double nnplusplus::rand_double (double from, double to)
{
  return from + rand () / (double)(RAND_MAX / to);
}
