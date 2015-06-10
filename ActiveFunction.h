#include <math.h>

namespace nnplusplus {

class ActiveFunction 
{
public:
	virtual double operator () (const double x) = 0;
};


class NullFunction: public ActiveFunction
{
public:
	double operator () (const double x) {
		printf ("null function\n");
		return x;
	}
};


class NegationFunction: public ActiveFunction
{
public:
	double operator () (const double x) {
		return -x;
	}
};

class LogisticSigmodFunction: public ActiveFunction
{
public:
	double operator () (const double x) {
		return 1 / (1 + exp (-x));
	}
};

class TanhFunction: public ActiveFunction
{
public:
	double operator () (const double x) {
		double e1 = exp (x), e2 = exp (-x);
		return (e1 - e2) / (e1 + e2);
	}
};

}
