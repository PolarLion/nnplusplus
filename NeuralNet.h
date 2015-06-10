#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "ActiveFunction.h"



namespace nnplusplus {


class NeuralNet
{
	const int layer_num;
public:
	//all weights 
	std::vector<double> weights;
	//every layer's size
	std::vector<int> layer_size;
	//every layer's active function
	std::vector<ActiveFunction*> active_function;
	//every layer's basis
	std::vector<double> basis;

	NeuralNet () : layer_num (0) {}
	NeuralNet (int layer_num, ...);

	bool init_weights ();
	bool init_basis ();
	bool propagation (const std::vector<double>& x, std::vector<double>& output, const int layer);
	bool output (const std::vector<double>& input, std::vector<double>& out);
	bool sum_of_squares_error (const std::vector<double>& x, const std::vector<double>& t, double& error);
	bool train ();
	

	void test ();
};


}




















