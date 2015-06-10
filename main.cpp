#include <iostream>

#include "NeuralNet.h"

using namespace nnplusplus;


int main (int argc, char** argv) 
{
	std::cout << "hello world\n";
	NeuralNet n (3, 5, 2, 5, "negation", "logistic");
	//n.init_weights ();
	n.test ();
	return 0;
}
