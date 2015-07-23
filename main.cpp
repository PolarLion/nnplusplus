#include <iostream>
//#include <omp.h>

#include "NeuralNet.h"
#include "auxiliary.h"

using namespace nnplusplus;
using namespace std;

int main (int argc, char** argv) 
{
  //omp_set_num_threads(32);
  //Eigen::setNbThreads(32);
  //Eigen::initParallel();
	std::cout << "hello world\n";
	
	//char* pend = NULL;
	int epoch = 1;
	double learning_rate = 1;
	if (2 == argc) {
		epoch = strtol (argv[1], NULL, 10);
	}
	if (3 == argc) {
		epoch = strtol (argv[1], NULL, 10);
		learning_rate = strtod (argv[2], NULL);
	}
	//NeuralNet n ("test/model.txt");
	NeuralNet n (epoch, learning_rate, 4, 2, 10, 10, 1, "logistic", "logistic", "logistic");
	//NeuralNet n (epoch, learning_rate, 3, 2, 2, 1, "logistic", "logistic");
	//NeuralNet n (epoch, learning_rate, 3, 2, 2, 1, "tanh", "tanh");
	n.show ();

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	n.test ();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  std::cout << "paramter nubmer " << n.paramter_number() << std::endl;
  std::cout << "propagation time: " << time_span.count() << " seconds" << std::endl;
  
	//n.load ("test/model.txt");
  //return 0;
  /*
	std::vector <double> x(2);
	std::vector <double> out;
	x [0] = 1;
	x [1] = 0;
	n.output (x, out);
	cout << "input 1 0 \n";
	for (int i = 0; i < out.size (); ++i) {
		std::cout << "out : " << out [i] << std::endl;
	}
	x [0] = 1;
	x [1] = 1;
	n.output (x, out);
	cout << "input 1 1\n";
	for (int i = 0; i < out.size (); ++i) {
		cout << "out : " << out[i] << endl;
	}
  out.clear();
  start = std::chrono::steady_clock::now();
  n.propagation (x, out);
  end = std::chrono::steady_clock::now();
  time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  std::cout << "propagation time: " << time_span.count() << " seconds" << std::endl;
  std::cout << "result " << out[out.size()-1] << std::endl;
  //n.output (x, out);
  //std::cout << "result " << out[0] << std::endl;
  */ 
  /*
  Eigen::VectorXd xx(2);
  xx[0] = 1;
  xx[1] = 1;
  std::vector<Eigen::VectorXd> xout;
  //start = std::chrono::steady_clock::now();
  n.propagation (xx, xout);
  //end = std::chrono::steady_clock::now();
  //time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  //std::cout << "new propagation time: " << time_span.count() << " seconds" << std::endl;
  std::cout << "input " << xout[0] << std::endl;
  std::cout << "result " << xout[xout.size()-1] << std::endl;
  xx[0] = 1;
  xx[1] = 0;
  std::vector<Eigen::VectorXd> xout1;
  n.propagation (xx, xout1);
  std::cout << "input " << xout1[0] << std::endl;
  std::cout << "result " << xout1[xout1.size()-1] << std::endl;
  */
  std::vector<Eigen::Vector2d> vx(4);
  vx[0] << 1, 0;
  vx[1] << 0, 1;
  vx[3] << 0, 0;
  vx[2] << 1, 1;
  for (int i = 0; i < 4; ++i) {
    std::vector<Eigen::VectorXd> xout;
    n.propagation (vx[i], xout);
    std::cout << "input " << xout[0] << std::endl;
    std::cout << "result " << xout[xout.size()-1] << std::endl;
  }
  
	return 0;
}
