#include <math.h>
#include <string>

namespace nnplusplus {

class ActiveFunction 
{
public:
	virtual std::string name () const = 0;
	virtual double operator () (const double x) = 0;
};


class NullFunction: public ActiveFunction
{
public:
	std::string name () const {
		return "null";
	}

	double operator () (const double x) {
		printf ("null function\n" );
		return x;
	}
};


class LogisticSigmodFunction: public ActiveFunction
{
public:
	std::string name () const {
		return "logisticsigmod";
	}

	double operator () (const double x) {
		return 1 / (1 + exp (-x));
	}
};

class TanhFunction: public ActiveFunction
{
public:
	std::string name () const {
		return "tanh";
	}

	double operator () (const double x) {
		double e1 = exp (x), e2 = exp (-x);
		return (e1 - e2) / (e1 + e2);
	}
};

}
