#include <iostream>
//#include <omp.h>
#include "Eigen/Core"
#include "NeuralNet.h"
#include "auxiliary.h"
#include <omp.h>

using namespace nnplusplus;
using namespace std;

int main (int argc, char** argv) 
{
	std::cout << "hello world\n";
	
	//char* pend = NULL;
  Eigen::initParallel();
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
	//NeuralNet n (epoch, learning_rate, 4, 2, 1000, 10000, 1, "logistic", "logistic", "logistic");
	NeuralNet n (epoch, learning_rate, 3, 2, 3, 1, "logistic", "logistic");
	//NeuralNet n (epoch, learning_rate, 3, 2, 2, 1, "tanh", "tanh");
	n.show ();

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
	n.test ();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end-start);
  //std::cout << "paramter nubmer " << n.paramter_number() << std::endl;
  std::cout << Eigen::nbThreads() << std::endl;
  std::cout << "test time: " << time_span.count() << " seconds" << std::endl;
  
	std::cout << "threads number " << Eigen::nbThreads() << std::endl;
  std::vector<Eigen::Vector2d> vx(4);
  vx[0] << 1, 0;
  vx[1] << 0, 1;
  vx[3] << 0, 0;
  vx[2] << 1, 1;
  for (int i = 0; i < 4; ++i) {
    //std::vector<Eigen::VectorXd> xout;
    n.propagation (vx[i]);
    std::cout << "input " << n.layer_out[0] << std::endl;
    std::cout << "result " << n.layer_out[n.layer_out.size()-1] << std::endl;
  }
  
	return 0;
}
