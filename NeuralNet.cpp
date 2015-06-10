#include "NeuralNet.h"
#include <string.h>
#include <math.h>


using namespace nnplusplus;


ActiveFunction* activefunction_maker (const char *str)
{
	ActiveFunction* p = NULL;
	printf ("str = %s\n", str);
	std::cout << strcmp ("negation", "negation") << std::endl;
	if (0 == strcmp ("negation", str)) {
		p = new NegationFunction ();
		if (NULL == p) {
			printf ("can't allocate memory for Negation Function\n");
			return NULL;
		}
		printf ("negation\n");
	}
	else if (0 == strcmp ("logistic", str)) {
		p = new LogisticSigmodFunction ();
		if (NULL == p) {
			printf ("Can't allocate memory for logistic function\n");
			return NULL;
		}
		printf ("logistic\n");
	}
	else {
		p = new NullFunction ();
		if (NULL == p) {
			printf ("Can't allocate memory for null function\n");
			return NULL;
		}
	}
	return p;
}


NeuralNet::NeuralNet (int ln, ...) 
	: layer_num (ln)
	, epoch(10)
	, learing_rate(0.5)
{
	va_list args;
	va_start (args, ln);
	for (int i = 0; i < layer_num; ++i) {
		layer_size.push_back (va_arg (args, int));
	}
	for (int i = 0; i < layer_num - 1; ++i) {
		active_function.push_back (activefunction_maker (va_arg (args, char*)));
	}
	va_end (args);
	for (int i = 0; i < layer_size.size (); ++i) {
		std::cout << layer_size [i] << " ";
	}
	std::cout << std::endl;
	std::cout << "active function : " << active_function.size () << std::endl;
	init_weights ();
	init_basis ();
}


bool NeuralNet::init_weights ()
{
 	int weight_num = 0;
	weights.clear ();
	for (int i = 0; i < layer_size.size () - 1; ++i) {
		weight_num += layer_size [i] * layer_size [i+1];
	}
	std::cout << "weight number : " << weight_num << std::endl;
	double w0 = 1.0 / weight_num;
	for (long i = 0; i < weight_num; ++i) {
		weights.push_back (w0);
	}
	std::cout << "weight size : " << weights.size () << std::endl;
	return true;
}


bool NeuralNet::init_basis ()
{
	for (int i = 0; i < layer_num - 1; ++i) {
		basis.push_back (1.0);
	}
	return true;
}


bool NeuralNet::propagation (const std::vector<double>& input, std::vector<double>& output, const int layer)
{
	output.clear ();
	if (input.size () != layer_size [layer]) {
		printf ("wrong input size");
		return false;
	}
 	if (layer >= layer_size.size () - 1 ) {
		printf ("wrong layer number\n");
		return false;
	}
	//std::cout << active_function.size () << std::endl;
	int base = 0;
	for (int i = 0; i < layer && i < layer_size.size () - 1; ++i) {
		base += layer_size [i] * layer_size [i+1];
	}
	for (int i = 0; i < layer_size [layer+1]; ++i) {
		double ne = 0;
		for (int j = 0; j < layer_size [layer]; ++j) {
			ne += input [j] * weights [base + i * input.size () + j];
		}
		ne += basis [layer];
		output.push_back ((*active_function [layer])(ne));
	}
 	return true;	
}


bool NeuralNet::output (const std::vector<double>& x, std::vector<double>& out)
{
	std::vector<double> tmp = x;
	for (int i = 0; i < layer_num - 1; ++i) {
		propagation (tmp, out, i);
		tmp = out;
	}
 	return true;
}


bool NeuralNet::sum_of_squares_error (const std::vector<double>& x, const std::vector<double>& t, double& error)
{
	if (t.size () != layer_size [layer_num-1]) {
		printf ("wrong target output size\n");
		return false;
	}
	std::vector<double> out;
	output (x, out);
	for (int i = 0; i < t.size (); ++i) {
		error += pow (t[i] - out[i], 2);
	}
	error /= 2;
	return true;
}


bool NeuralNet:load_training_set (const std::string& train_file, std::vector<std::pair<std::vector<double>, std::vector<double>>>) 
{
	return true;
}


bool NeuralNet::train_step (const std::vector<double>& x, const std::vector<double>& t)
{
	return true;
}


bool NeuralNet::train ()
{	
	std::vector<std::pair<std::vector<double>, std::vector<double>>> training_set;
	for (int i = 0; i < epoch; ++i) {
		for (int ii = 0; ii < training_set.size (); ++ii) {
			train_step (training_set[ii].first, training_set[ii].second);
		}
	}
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
	vector<double> out;
	output (x, out);
	for (int i = 0; i < out.size (); ++i) {
		cout << out[i] << " ";
	}
	cout << endl;
}



