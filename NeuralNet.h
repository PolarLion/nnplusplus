

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>



namespace nnplusplus {

class ActiveFunction 
{
public:
	double operator () (const double x) {
		return -x;
	}
};

class NeuralNet
{
	const int layer_num;
public:
	//all weights 
	std::vector<double> weights;
	//every layer's size
	std::vector<int> layer_size;
	//every layer's active function
	std::vector<ActiveFunction> active_function;

	NeuralNet () : layer_num (0) {}
	NeuralNet (int layer_num, ...);

	bool init_weights ();
	bool propagation (const std::vector<double>& x, std::vector<double>& output, const int layer);
	bool output (const std::vector<double>& input, std::vector<double>& out);
	

	void test ();
};


}
