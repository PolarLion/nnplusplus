#include "NeuralNet.h"
#include "auxiliary.h"
#include <string.h>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include "Eigen/Core"


using namespace nnplusplus;

NeuralNet::NeuralNet ( const std::string& model_filename)
  : epoch (0)
{
  load (model_filename);
}

NeuralNet::NeuralNet (int epoch_num, double learningrate, int ln, ...) 
  : epoch(epoch_num)
  , learing_rate(learningrate)
  , layer_num (ln)
{
  va_list args;
  va_start (args, ln);
  for (int i = 0; i < layer_num; ++i) {
    layer_size.push_back (va_arg (args, int));
  }
  if (layer_size.size() > 1) {
    input_num = layer_size[0];
    output_num = layer_size[layer_num-1];
  }
  for (int i = 0; i < layer_num - 1; ++i) {
    active_function.push_back (activefunction_maker (va_arg (args, char*)));
  }
  va_end (args);
  init_weight ();
  init_bias_weight();
  init_biasv ();
  init_layer_out();
  init_layer_delta();
}

NeuralNet::~NeuralNet ()
{
  clear();
}


bool NeuralNet::init_layer_out() 
{
  for (long i = 0; i < layer_num; ++i) {
    Eigen::VectorXd v = Eigen::VectorXd::Constant(layer_size[i], 0.0);
    layer_out.push_back(v);
  }
  return true;
}

bool NeuralNet::init_layer_delta()
{
  for (long layer = layer_num-1; layer > 0; --layer) {
    Eigen::VectorXd v = Eigen::VectorXd::Constant(layer_size[layer], 0.0);
    layer_delta.push_back(v);
  }
  return true;
}

//Eigen
bool NeuralNet::init_biasv ()
{
  for (int i = 1; i < layer_num; ++i) {
    //Eigen::VectorXd v = Eigen::VectorXd::Constant(layer_size[i], -1.0);
    biasv.push_back(-1.0);
    //std::cout << v << std::endl;
  }
  return true;
}

//Eigen
bool NeuralNet::init_weight ()
{
  for (unsigned long i = 0; i < layer_size.size() - 1; ++i) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(layer_size[i+1], layer_size[i]); 
    //std::cout << m << std::endl;
    weight.push_back(m);
  }
  return true;
}

//Eigen
bool NeuralNet::init_bias_weight()
{
  for (unsigned long i = 1; i < layer_num; ++i) {
    Eigen::VectorXd v = Eigen::VectorXd::Constant(layer_size[i], 0.1);
    bias_weight.push_back(v);
  }
  return true;
}

//Eigen
bool NeuralNet::load_training_set (const std::string& train_file, std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>& training_set)
{
  std::ifstream infile (train_file);
  if (infile.fail ()) {
    printf ("euralNet::load_training_set open file %s error", train_file.c_str ());
    return false;
  }
  std::string line;
  char* pend = NULL;
  std::getline (infile, line);
  unsigned long training_size = strtol (line.c_str (), &pend, 10);
  //std::cout << "training size : " << training_size << std::endl;
  input_num = strtol (pend, &pend, 10);
  //std::cout << "input : " << input_num << std::endl;
  output_num = strtol (pend, NULL, 10);
  //std::cout << "output : " << output_num << std::endl;
  while (training_set.size () < training_size && !infile.eof ()) {
    Eigen::VectorXd in(input_num);
    Eigen::VectorXd out(output_num);
    std::getline (infile, line);
    if (input_num < 2) {
      in[0] =  strtod (line.c_str (), NULL);
    }
    else {
      in[0] = strtod (line.c_str (), &pend);
      for (int i = 1; i < input_num - 1; ++i) {
        in[i] = strtod (pend, &pend);
      }
      in[input_num-1] = strtod (pend, NULL);
    }
    std::getline (infile, line);
    if (output_num < 2) {
      out[0] = strtod (line.c_str (), NULL);
    }
    else {
      out[0] = strtod (line.c_str (), &pend);
      for (int i = 1; i < output_num - 1; ++i) {
        out[i] = strtod (pend, &pend);
      }
      out[output_num-1] = strtod (pend, NULL);
    }
    training_set.push_back (std::pair<Eigen::VectorXd, Eigen::VectorXd>(in, out));
  }
  if (training_set.size () < training_size) {
    printf ("NeuralNet::load_training_set less of training set\n");
    return false;
  }
  infile.close ();
  /*
  for (auto p = training_set.begin(); p != training_set.end(); ++p) {
    std::cout << p->first << std::endl << std::endl << p->second << std::endl << std::endl;
  }
  */
  return true;
}

//Eigen
bool NeuralNet::sum_of_squares_error (const std::vector<Eigen::VectorXd>& layer_out, const Eigen::VectorXd& t, double& error)
{
  //printf ("NeuralNet::sum_of_squares_error\n");
  for (unsigned long i = 1; i <= t.size(); ++i) {
    error += pow (t[t.size () - i] - layer_out[layer_num-1][t.size () - i], 2);
    //std::cout << "index " << t.size()-i << " t " << t[t.size () - i] << " out " << layer_out[layer_num-1][t.size () - i] << " error " << error << std::endl;
  }
  error /= 2;
  return true;
}

//Eigen
bool NeuralNet::propagation (const Eigen::VectorXd& x)
{
  //std::cout << "NeuralNet::propagation " <<  Eigen::nbThreads() << std::endl;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

  layer_out[0] = x;
  //layer_out.push_back(x);
  for (unsigned long i = 0; i < weight.size(); ++i) {
    //Eigen::VectorXd v = (*active_function[i])((biasv[i] + weight[i] * layer_out[i]));
    //layer_out.push_back(v);
    layer_out[i+1] = (*active_function[i])((bias_weight[i]*biasv[i] + weight[i] * layer_out[i]));
  }
  //printf ("NeuralNet::propagation end\n");
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  //std::cout << "propagation time: " << time_span.count() << " seconds" << std::endl;
  return true;
}

//Eigen
bool NeuralNet::output (const Eigen::VectorXd& x, Eigen::VectorXd& out)
{
  std::vector<Eigen::VectorXd> o;
  propagation (x);
  out = o[o.size()-1];
  return true;
}

//Eigen
bool NeuralNet::compute_delta (const Eigen::VectorXd&t, const std::vector<Eigen::VectorXd>& layer_out, std::vector<Eigen::VectorXd>& layer_delta)
{
  //printf ("euralNet::compute_delta\n");
  //输出层
  //Eigen::VectorXd v = Eigen::VectorXd::Constant(layer_size[layer_num-1], 0.0);
  for (long i = 0; i < layer_size[layer_num-1]; ++i) {
    double o = layer_out[layer_num-1][i];
    //v[i] = t[i] - o;
    layer_delta[0][i] = t[i] - o;
    if ("logistic" == active_function[active_function.size() - 1]->name()) {
      //std::cout << "sigmod\n";
      //v[i] *= (o - pow (o, 2));  
      layer_delta[0][i] *= (o - pow (o, 2));
    }
    else if ("tanh" == active_function[active_function.size() - 1]->name()) {
      //v[i] *= 1 - pow (o, 2);
      layer_delta[0][i] *= (1 - pow (o, 2));
    }
  }
  //layer_delta.push_back(v);

  //隐层
  for (long layer = layer_num-2; layer > 0; --layer) {
    Eigen::VectorXd vh = Eigen::VectorXd::Constant(layer_size[layer], 0.0);
		//std::cout << "vh " << vh.size() << std::endl;
		//std::cout << "ld " << layer_delta[layer_num-1-layer].size () << std::endl;
    for (long i = 0; i < layer_size[layer]; ++i){
      layer_delta[layer_num-1-layer][i] = 0.0;
      for (long ii = 0; ii < layer_size[layer+1]; ++ii) {
        //vh[i] += layer_delta[layer_num-1-(layer+1)][ii] * weight[layer](ii, i); 
        layer_delta[layer_num-1-layer][i] += layer_delta[layer_num-1-(layer+1)][ii] * weight[layer](ii, i); 
      }
      double o = layer_out[layer][i];
      if ("logistic" == active_function[layer-1]->name()) {
        //printf ("log\n");
        //vh[i] *= (o - pow(o, 2));
        layer_delta[layer_num-1-layer][i] *= (o - pow(o, 2));
      }
      else if ("tanh" == active_function[layer-1]->name()) {
        //vh[i] *= 1 - pow(o, 2);
        layer_delta[layer_num-1-layer][i] *=1 - pow(o, 2);
      }
    }
    //layer_delta.push_back(vh);
  }
  //printf ("enenenenen\n");
  return true;
}

//Eigen
bool NeuralNet::update_weights (const Eigen::VectorXd& t, const std::vector<Eigen::VectorXd>& layer_out)
{
  //printf ("NeuralNet::update_weights\n");
  //std::cout << "weight size " << weight.size() << std::endl;
  //std::vector<Eigen::VectorXd> layer_delta;
  compute_delta (t, layer_out, layer_delta);
  
  //printf ("start update weight\n");
  
  for (long layer = layer_num-2; layer >= 0; --layer) {
    //std::cout << "layer " << layer << std::endl;
    //std::cout << "out \n" << layer_out[layer] << std::endl;
    //std::cout << "delat\n" << layer_delta[layer_num-2-layer] << std::endl;
    weight[layer] += learing_rate * layer_delta[layer_num-2-layer] * layer_out[layer].transpose();
    //for (long i = 0; i < layer_size[layer]; ++i) {
      //for (long ii = 0; ii < layer_size[layer+1]; ++ii) {
        //std::cout << "ii " << ii << " i " << i << std::endl;
        //std::cout << "delta index " << layer_num-2-layer << std::endl;
        //std::cout << layer_out[layer][i] << " " << layer_delta[layer_num-2-layer][ii] << std::endl;
        //weight[layer](ii, i) += learing_rate * layer_out[layer][i] * layer_delta[layer_num-2-layer][ii];
        //std::cout << learing_rate * layer_out[layer][i] * layer_delta[layer_num-2-layer][ii] << std::endl;
      //}
    //}
    for (long ii = 0; ii < layer_size[layer+1]; ++ii) {
      //std::cout << "biasv delta index " << layer_num-2-layer << std::endl;
      bias_weight[layer][ii] += learing_rate * (-1) * layer_delta[layer_num-2-layer][ii];
    }
  }
  return true;
}

//Eigen
bool NeuralNet::train_step (double& e, const Eigen::VectorXd& x, const Eigen::VectorXd& t)
{
  //printf ("NeuralNet::train_step\n");
  //std::vector<Eigen::VectorXd> layer_out;
  propagation (x);
  //std::cout << layer_out[0] << std::endl;
  double error = 0;
  sum_of_squares_error (layer_out, t, error);
  e += error;
  update_weights (t, layer_out);
  return true;
}

//Eigen
bool NeuralNet::train (const std::string& train_file)
{
  std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> training_set;
  load_training_set (train_file, training_set);
  int i = 0;
  double e = 0;
  for (i = 0; i < epoch; ++i) {
    shuffle (training_set);
    e = 0;
    for (unsigned long ii = 0; ii < training_set.size(); ++ii) {
      train_step (e, training_set[ii].first, training_set[ii].second);
    }
    if (e < 0.00001) {
      //printf("NeuralNet::train () : after %d epoches, error = %f, learning rate = %f\n\n", i, e, learing_rate);
      break;
    }
  }
  printf("NeuralNet::train () : after %d epoches, error = %f, learning rate = %f\n\n", i, e, learing_rate);
  return true;
}

bool NeuralNet::clear ()
{
  layer_num = 0;
  input_num = 0;
  output_num = 0;
  weight.clear ();
  layer_size.clear ();
  biasv.clear ();
  for (unsigned long i = 0; i < active_function.size (); ++i) {
    if (NULL != active_function [i]) {
      delete active_function [i];
    }
  }
  active_function.clear();
  return true;
}

//Eigen
bool NeuralNet::save (const std::string& model_file)
{
  std::ofstream outfile (model_file);
  if (outfile.fail ()) {
    printf ("NeuralNet::save : open file error %s/n", model_file.c_str ());
    return false;
  }
  outfile << layer_num << std::endl;
  for (unsigned long i = 0; i < layer_size.size (); ++i) {
    outfile << layer_size [i] << " ";
  }
  outfile << std::endl;
  for (unsigned long i = 0; i < active_function.size (); ++i) {
    outfile << active_function [i]->name () << " ";
  }
  outfile << std::endl;
  for (unsigned long i = 0; i< biasv.size (); ++i) {
    outfile << biasv[i] << " ";
  }
  outfile << std::endl;
  for (unsigned long i = 0; i < weight.size (); ++i) {
    outfile << weight[i] << std::endl;
  }
  outfile << std::endl;
  for (unsigned long i = 0; i < bias_weight.size(); ++i) {
    outfile << bias_weight[i] << std::endl;
  }
  outfile << std::endl;
  return true;
}

//Eigen
bool NeuralNet::load (const std::string& model_file)
{
  clear();
  std::ifstream infile (model_file);
  if (infile.fail ()) {
    printf ("NeuralNet::load () : open file %s error\n", model_file.c_str ());   
    return false;
  }
  infile >> layer_num;
  for (int i = 0; i < layer_num; ++i) {
    int n = 0;
    infile >> n;
    layer_size.push_back (n);
  }
  input_num = layer_size [0];
  output_num = layer_size [layer_num-1];
  std::string fun_name;
  for (int i = 0; i < layer_num-1; ++i) {
    infile >> fun_name;
    active_function.push_back (activefunction_maker (fun_name.c_str ()));
  }
  for (int i = 0; i < layer_num - 1; ++i) {
    double n = 0;
    infile >> n;
    biasv.push_back (n);
  }
  init_weight();
  for (long layer = 0; layer < layer_num-1; ++layer) {
    for (long ii = 0; ii < layer_size[layer+1]; ++ii) {
      for (long iii = 0; iii < layer_size[layer]; ++iii) {
        double d = 0.0;
        infile >> d;
        weight[layer](ii, iii) = d;
      }
    }
    //std::cout << weight[layer] << std::endl;
  }
  init_bias_weight();
  for (long layer = 1; layer < layer_num; ++layer) {
    for (long i = 0; i < layer_size[layer]; ++i) {
      double d = 0.0;
      infile >> d;
      bias_weight[layer-1][i] = d;
    }
    //std::cout << bias_weight[layer-1] << std::endl;
  }
  return true;
}


void NeuralNet::show () const
{
  std::cout << "layer number : " << layer_num << std::endl;
  std::cout << "layers size" << std::endl;
  for (unsigned long i = 0; i < layer_size.size (); ++i) {
    std::cout << layer_size [i] << " ";
  }
  std::cout << std::endl;
  std::cout << "active functions " << std::endl;
  for (unsigned long i = 0; i < active_function.size (); ++i) {
    std::cout << active_function [i]->name () << " ";
  }
  std::cout << std::endl;
  std::cout << "weight size " << weight.size() << std::endl;
  /*
  std::cout << "biaes " << std::endl;
  for (unsigned long i = 0; i< bias.size (); ++i) {
    std::cout << bias [i] << " ";
  }
  std::cout << std::endl;

  std::cout << "weights " << std::endl;
  std::cout << weights.size () << std::endl;
  for (unsigned long i = 0; i < weights.size (); ++i) {
    std::cout << weights [i] << " ";
  }
  std::cout << std::endl;
  */
}

long NeuralNet::paramter_number() const
{
  long n = 0;
  for (long i = 0; i < weight.size(); ++i) {
    n += weight[i].size();
  }
  for (long i = 0; i < bias_weight.size(); ++i) {
    n += bias_weight[i].size();
  }
  return n;
}

void NeuralNet::test ()
{ 
  Eigen::initParallel();
  //std::vector<std::pair<std::vector<double>, std::vector<double>>> t;
  //std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>> t;
  //load_training_set ("test/train.txt", t);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  train ("test/train.txt");
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  std::cout << "parameters number: " << paramter_number() << std::endl;
  std::cout << "training time: " << time_span.count() << " seconds" << std::endl;
  //save ("test/model.txt");
  //load ("test/model.txt");
}

/*
bool NeuralNet::init_bias ()
{
  for (int i = 0; i < layer_num-1; ++i) {
    bias.push_back (-1.0);
  }
  //std::cout << "bias size : " << bias.size () << std::endl;
  return true;
}

bool NeuralNet::init_weights ()
{
  int weight_num = 0;
  weights.clear ();
  //每一层加上一个bias的权重
  for (unsigned long i = 0; i < layer_size.size () - 1; ++i) {
    weight_num += (layer_size [i] + 1 ) * layer_size [i+1];
  }
  double w0 = 0.1;//1.0 / weight_num;
  for (long i = 0; i < weight_num; ++i) {
    weights.push_back (w0);
  }
  //std::cout << "weight size : " << weights.size () << std::endl;
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
  unsigned long training_size = strtol (line.c_str (), &pend, 10);
  //std::cout << "training size : " << training_size << std::endl;
  input_num = strtol (pend, &pend, 10);
  //std::cout << "input : " << input_num << std::endl;
  output_num = strtol (pend, NULL, 10);
  //std::cout << "output : " << output_num << std::endl;
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
  //for (int i = 0; i < training_set.size (); ++i) {
  //  for (int ii = 0; ii < training_set [i].first.size (); ++ii) {
  //    std::cout << training_set [i].first [ii] << " ";
  //  }
  //  std::cout << " : ";
  //  for (int ii = 0; ii < training_set [i].second.size (); ++ii) {
  //    std::cout << training_set [i].second [ii] << " ";
  //  }
  //  std::cout << std::endl;
  //}
  //std::cout << std::endl;
  return true;
}

bool NeuralNet::sum_of_squares_error (const std::vector<double>& out, const std::vector<double>& t, double& error)
{
  if (t.size () != layer_size [layer_num-1]) {
    printf ("NeuralNet::sum_of_squares_error() : wrong target output size\n");
     return false;
  }
  for (unsigned long i = 1; i <= t.size (); ++i) {
    error += pow (t[t.size () - i] - out[out.size () - i], 2);
  }
  error /= 2;
  return true;
}

bool NeuralNet::propagation (const std::vector<double>& x, std::vector<double>& out)
{
  out.clear(); 
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
        //std::cout <<  "   out : " << out [o_base + ii] << std::endl;
      }
      //std::cout << "bias weights : " << w_base + i * (1+layer_size [layer]) + layer_size [layer] << std::endl;
      ne += bias [layer] * weights [w_base + i * (1+layer_size [layer]) + layer_size [layer]];
      //std::cout << "bias " << bias [layer] << std::endl;
      //std::cout << ne << std::endl;
      out.push_back ((*active_function [layer])(ne));
    }
  }
  return true; 
}

bool NeuralNet::output (const std::vector<double>& x, std::vector<double>& out)
{
  out.clear ();
  std::vector<double> o;
  propagation (x, o);
  for (int i = 0; i < output_num; ++i) {
    out.push_back (o [o.size () - output_num + i]);
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
    if ("logistic" == active_function [active_function.size () - 1]->name ()) {
      //std::cout << "sigmod\n";
       delta [i] *= o - pow (o, 2);;  
    }
    else if ("tanh" == active_function [active_function.size () - 1]->name ()) {
      delta [i] *= 1 - pow (o, 2);
    }
  }

  //隐藏层
  int o_base = out.size () - layer_size [layer_num-1];
  int w_base = weights.size () - layer_size [layer_num-1] * (1+layer_size [layer_num-2]);
  int d_base = layer_size [layer_num - 1];
  for (int layer = layer_num - 2; layer > 0; --layer) {
    o_base -= layer_size [layer];
    for (int i = 0; i < layer_size [layer]; ++i) {
      delta.push_back (0.0);
      //std::cout << "oi " << o_base + i<< std::endl;
      double o = out [o_base + i];
      //std::cout << "out " << o_base.size () - 2 + i << " " <<  o << std::endl;
      int delta_index = d_base + i;
      //std::cout << "di "  << delta_index << std::endl;
      //delta[delta_index] = 0.0;
      for (int ii = 0; ii < layer_size [layer+1]; ++ii) {
        if (layer+1 == layer_num-1) {
          //std::cout << "wi " << w_base + i * layer_size [layer+1] + ii  << std::endl;
          delta [delta_index] += weights [w_base + i * layer_size [layer_num-1] + ii] * delta [ii];
        }
        else {
          //std::cout << "wi " << w_base + i * (1+layer_size [layer+1]) + ii  << std::endl;
          delta [delta_index] += weights [w_base + i * (1+layer_size [layer_num-1]) + ii] * delta [ii];
        }
        //std::cout << ii <<std::endl;
      }
      if ("logistic" == active_function [active_function.size () - 1]->name ()) {
        delta [delta_index] *= o * (1 - o); 
      }
      else if ("tanh" == active_function [active_function.size () - 1]->name ()) {
        delta [delta_index] *= 1 - pow (o, 2);
      }
      //std::cout << " delta " << delta_index << " " << delta[delta_index] << std::endl;
    }
    d_base += layer_size [layer];
    w_base -= layer_size [layer] * (1+layer_size [layer]);
  }
  //std::cout << "delta : ";
  //for (int i = 0; i < delta.size (); ++i) {
    //std::cout << delta [i] << " ";
  //}
  //std::cout << std::endl;

  return true;
}

bool NeuralNet::update_weights (const std::vector<double>& t, const std::vector<double>& out)
{
  if (t.size () <= 0) {
    printf ("NeuralNet::compute_local_gradient () : error target vector\n");
    return false;
  }
  //std::cout << "old weightx : ";
  //for (unsigned long i = 0; i < weights.size (); ++i) {
    //std::cout << weights [i] << " ";
  //}
  //std::cout << std::endl;
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
      //std::cout << learing_rate * delta [i] * out [o_base + ii] << std::endl;
      weights [w_base + i * (1 + layer_size [layer_num - 3]) + ii] += learing_rate * delta [i] * out [o_base + ii];
    }
    //bias
    //std::cout << " wi " << w_base + i * (1+layer_size [layer_num - 2]) + layer_size [layer_num - 2] << std::endl;
    //std::cout << " bi " << layer_num-2 << std::endl;
    //std::cout << " delta i " << i << " " << learing_rate * delta [i] * bias [layer_num-2] << std::endl;
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
        //std::cout << "delta i " << d_base + i << " " << learing_rate * delta [d_base + i] * out [o_base + ii] << std::endl;
        weights [w_base + i * (1 + layer_size [layer-1]) + ii] += learing_rate * delta [d_base + i] * out [o_base + ii];
      }
      //std::cout << "wi " << w_base + i * (1 + layer_size [layer-1]) + layer_size [layer - 1] << std::endl;
      //std::cout << bias [layer-1] << std::endl;
      //std::cout << learing_rate * delta [d_base + i] * bias [layer-1] << std::endl;
      weights [w_base + i * (1 + layer_size [layer-1]) + layer_size [layer - 1]] += learing_rate * delta [d_base + i] * bias [layer-1];
    }
  }

  //std::cout << "new weightx : ";
  //for (int i = 0; i < weights.size (); ++i) {
    //std::cout << weights [i] << " ";
  //}
  //std::cout<<std::endl;
  return true;
}

bool NeuralNet::train_step (double& e, const std::vector<double>& x, const std::vector<double>& t)
{
  //std::cout << x.size () << std::endl;
  
  //输入样本计算输出
  std::vector<double> out;
  propagation (x, out);
  //for (int ii = 0 ; ii < x.size (); ++ii) {
  //  std::cout << x[ii] << " ";
  //}
  //std::cout << " : ";
  //for (int ii = 0 ; ii < t.size (); ++ii) {
  //  std::cout << t[ii] << " ";
  //}
  //std::cout << " : ";
  //for (int ii = 0; ii < out.size (); ++ii) {
  //  std::cout << out[ii] << " ";
  //}  
  //std::cout << std::endl;
  //计算输出层误差
  double error = 0;
  sum_of_squares_error (out, t, error);
  e += error;
  //计算局部梯度并修正权值
  update_weights (t, out);  

  //std::cout << "error : " <<  error << std::endl << std::endl;
  return true;
}

bool NeuralNet::train (const std::string& train_file)
{  
  std::vector<std::pair<std::vector<double>, std::vector<double>>> training_set;
  load_training_set (train_file, training_set);
  int i = 0;
  //double e1 = 0;
  double e2 = 0;
  for (i = 0; i < epoch; ++i) {
    //对训练集和随机洗牌
    shuffle (training_set);
    //e1 = e2;
    e2 = 0;
    for (unsigned long ii = 0; ii < training_set.size (); ++ii) {
      train_step (e2, training_set[ii].first, training_set[ii].second);  
    }
    //printf("NeuralNet::train () : error = %f\n\n", e);
    //if (e2 / e1 < 0.9) {
      //learing_rate /= 2;
      //printf("NeuralNet::train () : after %d epoches, error = %f, learning rate = %f\n\n", i, e2, learing_rate);
    //}
    if (e2 < 0.000001) {
      printf("NeuralNet::train () : after %d epoches, error = %f, learning rate = %f\n\n", i, e2, learing_rate);
      break;
    }
  }
  printf("NeuralNet::train () : after %d epoches, error = %f, learning rate = %f\n\n", i, e2, learing_rate);
  return true;
}

bool NeuralNet::save (const std::string& model_file)
{
  std::ofstream outfile (model_file);
  if (outfile.fail ()) {
    printf ("NeuralNet::save : open file error %s/n", model_file.c_str ());
    return false;
  }
  //outfile << "layers num " << std::endl;
  outfile << layer_num << std::endl;
  //outfile << "layers size" << std::endl;
  for (unsigned long i = 0; i < layer_size.size (); ++i) {
    outfile << layer_size [i] << " ";
  }
  outfile << std::endl;
  //outfile << "active functions " << std::endl;
  for (unsigned long i = 0; i < active_function.size (); ++i) {
    outfile << active_function [i]->name () << " ";
  }
  outfile << std::endl;

  //outfile << "biaes " << std::endl;
  for (unsigned long i = 0; i< bias.size (); ++i) {
    outfile << bias [i] << " ";
  }
  outfile << std::endl;

  //outfile << "weights " << std::endl;
  outfile << weights.size () << std::endl;
  for (unsigned long i = 0; i < weights.size (); ++i) {
    outfile << weights [i] << " ";
  }
  outfile << std::endl;

  return true; 
}

*/
