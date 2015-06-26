#include <iostream>

#include "NeuralNet.h"
#include "auxiliary.h"

using namespace nnplusplus;
using namespace std;

int main (int argc, char** argv) 
{
	std::cout << "hello world\n";
	
	char* pend = NULL;
	int epoch = 1;
	double learning_rate = 1;
	if (2 == argc) {
		epoch = strtol (argv[1], NULL, 10);
	}
	if (3 == argc) {
		epoch = strtol (argv[1], NULL, 10);
		learning_rate = strtod (argv[2], NULL);
	}
	NeuralNet n (epoch, learning_rate, 3, 2, 2, 1, "logistic", "logistic");
	n.test ();
	return 0;

	vector<int> x;
	for (int i = 0; i < 10; ++i) {
		x.push_back (i);
		cout << rand_double (0, 2) << endl;
	}
	shuffle (x);
	for (int i = 0; i < x.size (); ++i) {
		cout << x[i] << " ";
	}
	cout << endl;
	return 0;
}
