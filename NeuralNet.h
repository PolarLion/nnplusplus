#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "ActiveFunction.h"



namespace nnplusplus {


class NeuralNet
{
private:
	const int epoch;
	double learing_rate;
	int layer_num;

	int training_size;
	int input_num;
	int output_num;
	ActiveFunctionMaker activefunction_maker;
public:
	//all weights 
	std::vector<double> weights;
	//every layer's size
	std::vector<int> layer_size;
	//every layer's active function
	std::vector<ActiveFunction*> active_function;
	//every layer's bias
	std::vector<double> bias;

	NeuralNet () :  epoch(0), learing_rate(0), layer_num (0) {}
	NeuralNet (const std::string& model_filename);
	NeuralNet (int epoch, double learingrate, int layer_num, ...);

	bool init_weights ();
	bool init_bias ();
	bool propagation (const std::vector<double>& x, std::vector<double>& output, const int layer);
	bool output (const std::vector<double>& input, std::vector<double>& out);
	bool sum_of_squares_error (const std::vector<double>& out, const std::vector<double>& t, double& error);
	bool load_training_set (const std::string& train_file, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_set);
	bool train_step (double& e, const std::vector<double>& x, const std::vector<double>& t);
	bool compute_delta (const std::vector<double>& t, const std::vector<double>& out, std::vector<double>& delta);
	bool update_weights (const std::vector<double>& t, const std::vector<double>& out);
	bool train (const std::string& train_file);
	bool save (const std::string& model_file);
	bool clear ();
	bool load (const std::string& model_file);
	void show () const;
	void test ();
};


}




















