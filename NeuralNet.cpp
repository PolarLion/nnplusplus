#include "NeuralNet.h"
#include "auxiliary.h"
#include <string.h>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <ctime>
#include <ratio>
#include <chrono>


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
  for (int i = 0; i < layer_num - 1; ++i) {
    active_function.push_back (activefunction_maker (va_arg (args, char*)));
  }
  va_end (args);
  init_weights ();
  init_bias ();
}

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
    weights.push_back (w0+i*0.01);
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
  /*
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
  */
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
        //std::cout <<  "   out : " << out [o_base + ii] << std::endl;
      }
      //std::cout << "bias weights : " << w_base + i * (1+layer_size [layer]) + layer_size [layer] << std::endl;
      ne += bias [layer] * weights [w_base + i * (1+layer_size [layer]) + layer_size [layer]];
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
    out.push_back (o [o.size () - 1 - output_num]);
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
  for (unsigned long i = 0; i < weights.size (); ++i) {
    //std::cout << weights [i] << " ";
  }
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
  double e1 = 0;
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

bool NeuralNet::clear ()
{
  layer_num = 0;
  input_num = 0;
  output_num = 0;
  weights.clear ();
  layer_size.clear ();
  bias.clear ();
  for (unsigned long i = 0; i < active_function.size (); ++i) {
    if (NULL != active_function [i]) {
      delete active_function [i];
    }
  }
  active_function.clear ();
  return true;
}

bool NeuralNet::load (const std::string& model_file)
{
  clear ();
  std::ifstream infile (model_file);
  if (infile.fail ()) {
    printf ("NeuralNet::load () : open file %s error\n", model_file.c_str ());
    return false;
  }
  //std::string comment_line;
  //std::getline (infile, comment_line);
  //std::cout << comment_line << std::endl;
  infile >> layer_num;
  //std::cout << "layer number " << layer_num << std::endl;
  //std::getline (infile, comment_line);
  //std::cout << comment_line << std::endl;
  for (int i = 0; i < layer_num; ++i) {
    int n = 0;
    infile >> n;
    layer_size.push_back (n);
    //std::cout << "layer size " << n << std::endl;
  }
  input_num = layer_size [0];
  input_num = layer_size [layer_num-1];
  //std::getline (infile, comment_line);
  //std::cout << comment_line << std::endl;
  std::string fun_name;
  for (int i = 0; i < layer_num-1; ++i) {
    infile >> fun_name;
    //std::cout << fun_name.c_str () << std::endl;
    active_function.push_back (activefunction_maker (fun_name.c_str ()));
  }
  //std::getline (infile, comment_line);
  //std::cout << comment_line << std::endl;
  for (int i = 0; i < layer_num - 1; ++i) {
    double n = 0;
    infile >> n;
    bias.push_back (n);
  }
  //std::getline (infile, comment_line);
  //std::cout << comment_line << std::endl;
  int weights_num = 0;
  infile >> weights_num;
  for (int i = 0; i < weights_num; ++i) {
    double n = 0;
    infile >> n;
    weights.push_back (n);
    //std::cout << "weight " << weights [i] << std::endl;
  }
  input_num = layer_size [0];
  output_num = layer_size [layer_size.size() - 1];
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
  std::cout << std::endl;*/
}

void NeuralNet::test ()
{
  using namespace std;
  //std::vector<std::pair<std::vector<double>, std::vector<double>>> t;
  //load_training_set ("test/train.txt", t);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  train ("test/train.txt");
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  std::cout << "parameters number: " << weights.size() << std::endl;
  std::cout << "training time: " << time_span.count() << " seconds" << std::endl;
  save ("test/model.txt");
  load ("test/model.txt");
  return;
}



