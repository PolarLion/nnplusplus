#include "NeuralNet.h"


using namespace nnplusplus;



NeuralNet::NeuralNet (int ln, ...) 
	: layer_num (ln)
{
	va_list args;
	va_start (args, ln);
	while (ln--) 
	{
		layer_size.push_back (va_arg (args, int));
	}
	va_end (args);
	for (int i = 0; i < layer_size.size (); ++i) {
		std::cout << layer_size [i] << " ";
	}
	printf ("\n");
}


bool NeuralNet::init_weights ()
{
 	int weight_num = 0;
	for (int i = 0; i < layer_size.size () - 1; ++i) {
		weight_num += layer_size [i] * layer_size [i+1];
	}
	std::cout << weight_num << std::endl;
	double w0 = 1.0 / weight_num;
	for (long i = 0; i < weight_num; ++i) {
		weights.push_back (w0);
	}
	return true;
}


bool NeuralNet::propagation (const std::vector<double>& input, std::vector<double>& output, const int layer)
{
 	if (layer >= layer_size.size () - 1 ) {
		printf ("wrong layer number\n");
		return false;
	}
	int base = 0;
	for (int i = 0; i < layer && i < layer_size.size () - 1; ++i) {
		base += layer_size [i] * layer_size [i+1];
	}
	for (int i = 0; i < layer_size [layer+1]; ++i) {
		double ne = 0;
		for (int j = 0; j < input.size (); ++j) {
			ne += input [j] * weights [base + i * input.size () + j];
		}
		output.push_back (active_function [layer](ne));
	}
 	return true;	
}


bool NeuralNet::output (const std::vector<double>& input, std::vector<double>& out)
{
	return true;
}



void NeuralNet::test ()
{
	using namespace std;
	vector<double> x;
	for (int i = 0; i < 5; ++i) {
		x.push_back (i/5.0);
		cout << x[i] << " ";
	}
	cout << endl;
	vector<double> h1;
	propagation (x, h1, 0);
	for (int i = 0; i < h1.size (); ++i) {
		cout << h1[i] << " ";
	}
	cout << endl;
}







