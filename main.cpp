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
	//n.test ();
	n.load ("test/model.txt");
	std::vector <double> x(2);
	std::vector <double> out;
	x [0] = 1;
	x [1] = 0;
	n.output (x, out);
	for (int i = 0; i < out.size (); ++i) {
		std::cout << "out : " << out [i] << std::endl;
	}
	return 0;

}
