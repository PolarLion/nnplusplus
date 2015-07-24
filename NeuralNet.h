#ifndef _NEURALNET_H_
#define _NEURALNET_H_

#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdarg.h>
#include "ActiveFunction.h"
#include "Eigen/Core"
#include "Eigen/Dense"


namespace nnplusplus {

class NeuralNet
{
private:
  const int epoch;
  double learing_rate;
  int layer_num;

  long input_num;
  long output_num;
  ActiveFunctionMaker activefunction_maker;
public:
  //all weights 
  std::vector<Eigen::MatrixXd> weight;
  std::vector<Eigen::VectorXd> bias_weight;
  //every layer's size
  std::vector<long> layer_size;
  //every layer's active function
  std::vector<ActiveFunction*> active_function;
  //matrix every layer's bias
  std::vector<double> biasv;

  NeuralNet () :  epoch(0), learing_rate(0), layer_num (0) {}
  NeuralNet (const std::string& model_filename);
  NeuralNet (int epoch, double learingrate, int layer_num, ...);
  ~NeuralNet ();

  //matrix
  bool init_weight ();

  //matrix
  bool init_bias_weight ();

  //matirx
  bool init_biasv ();

  //matrix
  bool propagation (const Eigen::VectorXd& X, std::vector<Eigen::VectorXd>& layer_out);

  //matrix
  bool output (const Eigen::VectorXd& x, Eigen::VectorXd& out);

  //matrix
  bool sum_of_squares_error (const std::vector<Eigen::VectorXd>& out, const Eigen::VectorXd& t, double& error);

  //matrix
  bool load_training_set (const std::string& train_file, std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_set);

  //matrix
  bool train_step (double& e, const Eigen::VectorXd& x, const Eigen::VectorXd& t);

  //matrix
  bool compute_delta (const Eigen::VectorXd&t, const std::vector<Eigen::VectorXd>& layer_out, std::vector<Eigen::VectorXd>& delta);

  //matrix
  bool update_weights (const Eigen::VectorXd& t, const std::vector<Eigen::VectorXd>& layer_out);

  //matrix
  bool train (const std::string& train_file);

  bool save (const std::string& model_file);
  bool clear ();
  bool load (const std::string& model_file);
  void show () const;
  long paramter_number() const;
  void test ();

  //std::vector<double> weights;
  //std::vector<double> bias;
  //bool init_bias ();
  //bool propagation (const std::vector<double>& input, std::vector<double>& layer_out);
  //bool output (const std::vector<double>& x, std::vector<double>& out);
  //bool sum_of_squares_error (const std::vector<double>& out, const std::vector<double>& t, double& error);
  //bool load_training_set (const std::string& train_file, std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_set);
  //bool train_step (double& e, const std::vector<double>& x, const std::vector<double>& t);
  //bool compute_delta (const std::vector<double>& t, const std::vector<double>& out, std::vector<double>& delta);
  //bool update_weights (const std::vector<double>& t, const std::vector<double>& layer_out);
  //bool init_weights ();
  //bool train (const std::string& train_file);
};

}


#endif
