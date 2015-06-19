#include <iostream>

#include "NeuralNet.h"
#include "auxiliary.h"

using namespace nnplusplus;
using namespace std;

int main (int argc, char** argv) 
{
	std::cout << "hello world\n";
	NeuralNet n (3, 5, 2, 5, "negation", "logistic");
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
