#include <iostream>

#include "NeuralNet.h"




using namespace nnplusplus;



int main (int argc, char** argv) 
{
	std::cout << "hello world\n";
	NeuralNet n (3, 5, 3, 0);
	n.init_weights ();
	n.test ();
	return 0;
}
