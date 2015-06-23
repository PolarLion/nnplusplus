#include "NeuralNet.h"
#include "auxiliary.h"
#include <string.h>
#include <math.h>
#include <fstream>
#include <stdio.h>


using namespace nnplusplus;


ActiveFunction* activefunction_maker (const char *str)
{
	ActiveFunction* p = NULL;
	printf ("str = %s\n", str);
	//std::cout << strcmp ("negation", "negation") << std::endl;
	if (0 == strcmp ("tanh", str)) {
		p = new TanhFunction ();
		if (NULL == p) {
			printf ("can't allocate memory for tanh Function\n");
			return NULL;
		}
		//printf ("negation\n");
	}
	else if (0 == strcmp ("logistic", str)) {
		p = new LogisticSigmodFunction ();
		if (NULL == p) {
			printf ("Can't allocate memory for logistic function\n");
			return NULL;
		}
		//printf ("logistic\n");
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
	, epoch(1)
	, learing_rate(1)
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
		//std::cout << layer_size [i] << " ";
	}
	//std::cout << std::endl;
	std::cout << "active function size : " << active_function.size () << std::endl;
	init_weights ();
	init_bias ();
}

bool NeuralNet::init_bias ()
{
	for (int i = 0; i < layer_num-1; ++i) {
		bias.push_back (-1.0);
	}
	std::cout << "bias size : " << bias.size () << std::endl;
	return true;
}

bool NeuralNet::init_weights ()
{
 	int weight_num = 0;
	weights.clear ();
	//每一层加上一个bias的权重
	for (int i = 0; i < layer_size.size () - 1; ++i) {
		weight_num += (layer_size [i] + 1 ) * layer_size [i+1];
	}
	double w0 = 0.1;//1.0 / weight_num;
	for (long i = 0; i < weight_num; ++i) {
		weights.push_back (w0);
	}
	std::cout << "weight size : " << weights.size () << std::endl;
	return true;
}

bool NeuralNet::load_training_set (const std::string& train_file, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_set) 
{
	std::ifstream infile (train_file);
	if (infile.fail ()) {
		printf ("euralNet::load_training_set open file %s error", train_file.c_str ());
		return false;
	}
	std::string line;
	char* pend = NULL;
	std::getline (infile, line);
	training_size = strtol (line.c_str (), &pend, 10);
	std::cout << "training size : " << training_size << std::endl;
	input_num = strtol (pend, &pend, 10);
	std::cout << "input : " << input_num << std::endl;
	output_num = strtol (pend, NULL, 10);
	std::cout << "output : " << output_num << std::endl;
	while (training_set.size () < training_size && !infile.eof ()) {
		std::vector <double> in;
		std::vector <double> out;
		std::getline (infile, line);
		if (input_num < 2) {
			in.push_back (strtod (line.c_str (), NULL));
		}
		else {
			in.push_back (strtod (line.c_str (), &pend));
			for (int i = 1; i < input_num - 1; ++i) {
				in.push_back (strtod (pend, &pend));
			}
			in.push_back (strtod (pend, NULL));
		}
		std::getline (infile, line);
		if (output_num < 2) {
			out.push_back (strtod (line.c_str (), NULL));
		}
		else {
			out.push_back (strtod (line.c_str (), &pend));
			for (int i = 1; i < output_num - 1; ++i) {
				out.push_back (strtod (pend, &pend));
			}
			out.push_back (strtod (pend, NULL));
		}
		training_set.push_back (std::pair<std::vector<double>, std::vector<double>>(in, out));
	}
	if (training_set.size () < training_size) {
		printf ("NeuralNet::load_training_set less of training set\n");
		return false;
	}
	infile.close ();
	for (int i = 0; i < training_set.size (); ++i) {
		for (int ii = 0; ii < training_set [i].first.size (); ++ii) {
			std::cout << training_set [i].first [ii] << " ";
		}
		std::cout << " : ";
		for (int ii = 0; ii < training_set [i].second.size (); ++ii) {
			std::cout << training_set [i].second [ii] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	return true;
}

bool NeuralNet::sum_of_squares_error (const std::vector<double>& out, const std::vector<double>& t, double& error)
{
	//std::cout << " XXXXXXXXXXX " << t.size () << " " << layer_size [layer_num-1] << std::endl;
	if (t.size () != layer_size [layer_num-1]) {
		printf ("NeuralNet::sum_of_squares_error() : wrong target output size\n");
		return false;
	}
	//std::vector<double> out;
	//output (x, out);
	for (int i = 1; i <= t.size (); ++i) {
		error += pow (t[t.size () - i] - out[out.size () - i], 2);
	}
	error /= 2;
	return true;
}


bool NeuralNet::output (const std::vector<double>& x, std::vector<double>& out)
{
	out.clear (); 
	out = x;
	int w_base = 0;
	int o_base = 0;
	for (int layer = 0; layer < layer_num - 1; ++layer) {
		if (layer > 0) {
			w_base += (layer_size [layer-1] + 1) * layer_size [layer];
			o_base += layer_size [layer-1];
		}
		for (int i = 0; i < layer_size [layer+1]; ++i) {
			double ne = 0;
			for (int ii = 0; ii < layer_size [layer]; ++ii) {
				//std::cout << " out index :" << o_base + ii ;
				//std::cout << "      weight index " << w_base + i * (1+layer_size [layer]) + ii;
				ne += out [o_base + ii] * weights [w_base + i * (1+layer_size [layer]) + ii]; 
			  //std::cout << std::endl;
			}
			//std::cout << "bias weights : " << w_base + i * (1+layer_size [layer]) + layer_size [layer] << std::endl;
			ne += bias [layer] * weights [w_base + i * (1+layer_size [layer]) + layer_size [layer]];
			out.push_back ((*active_function [layer])(ne));
		}
	}
 	return true; 
}


bool NeuralNet::compute_delta (const std::vector<double>& t, const std::vector<double>& out, std::vector<double>& delta)
{
	//输出层
	//std::cout << "layer size " << layer_size [layer_num-1] << std::endl;
	for (int i = 0; i < layer_size [layer_num-1]; ++i) {
	 	delta.push_back (0.0);
		double o = out [out.size () - layer_size [layer_num-1] + i];
		//std::cout << "out " << out.size () - layer_size [layer_num-1] + i << " " <<  o << std::endl;
		delta [i] = (t [i] - o);
		//std::cout << "delta " << delta [i] << std::endl;
		if ("logisticsigmod" == active_function [active_function.size () - 1]->name ()) {
			//std::cout << "sigmod\n";
	 		delta [i] *= o * (1 - o);	
		}
	}

	//隐藏层
	int o_base = out.size () - layer_size [layer_num-1];
	int w_base = weights.size () - layer_size [layer_num-1] * (1+layer_size [layer_num-2]);
	for (int layer = layer_num - 2; layer > 0; --layer) {
		o_base -= layer_size [layer];
		for (int i = 0; i < layer_size [layer]; ++i) {
			delta.push_back (0.0);
			//std::cout << o_base + i<< std::endl;
			double o = out [o_base + i];
			//std::cout << "out " << o_base.size () - 2 + i << " " <<  o << std::endl;
			int delta_index = layer_size [layer_num - 1] + i;
			//std::cout << "di "  << delta_index << std::endl;
			//delta[delta_index] = 0.0;
			for (int ii = 0; ii < layer_size [layer_num - 1]; ++ii) {
				//std::cout << "- " << w_base + i * layer_size [layer_num-1] + ii  << std::endl;
				//std::cout << ii <<std::endl;
				delta [delta_index] += weights [w_base + i * layer_size [layer_num-1] + ii] * delta [ii];
			}
			if ("logisticsigmod" == active_function [active_function.size () - 1]->name ()) {
				delta [delta_index] *= o * (1 - o); 
			}
			//std::cout << " delta " << delta_index << " " << delta[delta_index] << std::endl;
		}
	}
	std::cout << "delta : ";
	for (int i = 0; i < delta.size (); ++i) {
		std::cout << delta [i] << " ";
	}
	std::cout << std::endl;

	return true;
}


bool NeuralNet::update_weights (const std::vector<double>& t, const std::vector<double>& out)
{
	if (t.size () <= 0) {
		printf ("NeuralNet::compute_local_gradient () : error target vector\n");
		return false;
	}
	std::vector<double> delta;
	compute_delta (t, out, delta);
	//return false;
	
	int w_base = weights.size () - layer_size [layer_num - 1] * (1+layer_size [layer_num - 2]);
	int o_base = out.size () - layer_size [layer_num - 1] - layer_size [layer_num - 2];
	int d_base = 0;
	//输出层
	for (int i = 0; i < layer_size [layer_num - 1]; ++i) {
		for (int ii = 0; ii < layer_size [layer_num - 2]; ++ii) {
			//std::cout << " wi " << w_base + i * (1 + layer_size [layer_num - 2]) + ii << std::endl;
			//std::cout << " oi " << o_base + ii << std::endl;
			//std::cout << 
			weights [w_base + i * (1 + layer_size [layer_num - 3]) + ii] += learing_rate * delta [i] * out [o_base + ii];
		}
		//bias
		//std::cout << " wi " << w_base + i * (1+layer_size [layer_num - 2]) + layer_size [layer_num - 2] << std::endl;
		//std::cout << " bi " << layer_num-2 << std::endl;
		weights [w_base + i * (1+layer_size [layer_num - 1]) + layer_size [layer_num - 2]] += learing_rate * delta [i] * bias [layer_num-2];
	}

	//隐层
	for (int layer = layer_num - 2; layer > 0; --layer) {
		w_base -= layer_size [layer] * (1 + layer_size [layer - 1]);
		o_base -= layer_size [layer - 1];
		d_base += layer_size [layer + 1];
		for (int i = 0; i < layer_size [layer]; ++i) {
			for (int ii = 0; ii < layer_size [layer - 1]; ++ii) {
				//std::cout << "wi " << w_base + i * (1 + layer_size [layer-1]) + ii << std::endl;
				//std::cout << "oi " << o_base + ii << std::endl;
				//std::cout << "di " << d_base + i << std::endl;
				weights [w_base + i * (1 + layer_size [layer-1]) + ii] += learing_rate * delta [d_base + i] * out [o_base + ii];
			}
			//std::cout << w_base + i * (1 + layer_size [layer-1]) + layer_size [layer - 1] << std::endl;
			//std::cout << bias [layer-1] << std::endl;
			weights [w_base + i * (1 + layer_size [layer-1]) + layer_size [layer - 1]] += learing_rate * delta [d_base + i] * bias [layer-1];
		}
	}
	for (int i = 0; i < weights.size (); ++i) {
		//std::cout << weights [i] << " ";
	}
	std::cout<<std::endl;
	return true;
}


bool NeuralNet::train_step (double& e, const std::vector<double>& x, const std::vector<double>& t)
{
	//std::cout << x.size () << std::endl;
	
	//输入样本计算输出
	std::vector<double> out;
	output (x, out);
	for (int ii = 0 ; ii < x.size (); ++ii) {
		std::cout << x[ii] << " ";
	}
	std::cout << " : ";
	for (int ii = 0 ; ii < t.size (); ++ii) {
		std::cout << t[ii] << " ";
	}
	std::cout << " : ";
	for (int ii = 0; ii < out.size (); ++ii) {
		std::cout << out[ii] << " ";
	}	
	std::cout << std::endl;
	//计算输出层误差
	double error = 0;
	sum_of_squares_error (out, t, error);
	e += error;
	std::cout << "error : " <<  error << std::endl;
	//计算局部梯度并修正权值
	update_weights (t, out);	
	return true;
}


bool NeuralNet::train (const std::string& train_file)
{	
	std::vector<std::pair<std::vector<double>, std::vector<double>>> training_set;
	load_training_set (train_file, training_set);
	for (int i = 0; i < epoch; ++i) {
		//对训练集和随机洗牌
		shuffle (training_set);
		double e = 0;
		for (int ii = 0; ii < training_set.size (); ++ii) {
			train_step (e, training_set[ii].first, training_set[ii].second);	
		}
	 	printf("NeuralNet::train () : error = %f\n", e);
	}
	 return true;
}


void NeuralNet::test ()
{
	using namespace std;
	//std::vector<std::pair<std::vector<double>, std::vector<double>>> t;
	//load_training_set ("test/train.txt", t);
	train ("test/train.txt");
	return;
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
	cout  << endl;
}





/*
bool NeuralNet::propagation (const std::vector<double>& input, std::vector<double>& output, const int layer)
{
	if (input.size () != layer_size [layer]) { 
		std::cout << input.size () << " " << layer_size [layer] << std::endl; 
		printf ("NeuralNet::propagation() : wrong input size\n");
		return false;
	}
 	if (layer >= layer_size.size () - 1 ) {
		printf ("NeuralNet::propagation () : wrong layer number\n");
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
		//ne += basis [layer];
		output.push_back ((*active_function [layer])(ne));
	}
 	return true;	
}
*/
