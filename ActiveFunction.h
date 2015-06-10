#include <math.h>

class ActiveFunction 
{
public:
	virtual double operator () (const double x) = 0;
};


class NullFunction: public ActiveFunction
{
public:
	double operator () (const double x) {
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
