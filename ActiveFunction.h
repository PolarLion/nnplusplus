#ifndef _ACTIVEFUNCTION_H_
#define _ACTIVEFUNCTION_H_

#include <math.h>
#include <string.h> 
#include <string>
#include "Eigen/Dense"

namespace nnplusplus {

class ActiveFunction 
{
public:
  virtual std::string name () const = 0;
  virtual double operator () (const double x) = 0;
  virtual Eigen::VectorXd operator () (const Eigen::VectorXd& x) = 0;
};


class NullFunction: public ActiveFunction
{
public:
  std::string name () const {
    return "null";
  }

  double operator () (const double x) {
    printf ("null function\n" );
    return x;
  }

  Eigen::VectorXd operator () (const Eigen::VectorXd& x) {
    printf ("null function\n");
    return x;
  }
};


class LogisticSigmodFunction: public ActiveFunction
{
public:
  std::string name () const {
    return "logistic";
  }

  double operator () (const double x) {
    return 1 / (1 + exp (-x));
  }

  Eigen::VectorXd operator () (const Eigen::VectorXd& x) {
    //printf ("logistic\n");
    Eigen::VectorXd y = Eigen::VectorXd::Constant(x.size(), 0.0);
    for (auto i = 0; i < x.size(); ++i) {
      y[i] = 1 / (1 + exp (-x[i]));
    }
    return y;
  }
};

class TanhFunction: public ActiveFunction
{
public:
  std::string name () const {
    return "tanh";
  }

  double operator () (const double x) {
    double e1 = exp (x), e2 = exp (-x);
    return (e1 - e2)  / (e1 + e2);
  }

  Eigen::VectorXd operator () (const Eigen::VectorXd& x) {
    Eigen::VectorXd y = Eigen::VectorXd::Constant(x.size(), 0.0);
    for (auto i = 0; i < x.size(); ++i) {
      double e1 = exp (x[i]), e2 = exp (-x[i]);
      y[i] = (e1 - e2)  / (e1 + e2);
    }
    return y;
  }
};


class ActiveFunctionMaker 
{
public:
  ActiveFunction* operator () (const char* str) {
    ActiveFunction *p = NULL;
    if (0 == strcmp ("tanh", str)) {
      p = new TanhFunction ();
      if (NULL == p) {
        printf ("can't allocate memory for tanh Function\n");
        return NULL;
      }
    }
    else if (0 == strcmp ("logistic", str)) {
      p = new LogisticSigmodFunction ();
      if (NULL == p) {
        printf ("Can't allocate memory for logistic function\n");
        return NULL;
      }
    }
    else {
      p = new NullFunction ();
      if (NULL == p) {
        printf ("ActiveFunctionMaker: can't allocate memory for null function\n");
        return NULL;
      }
    }
    return p;
  }
};


}

#endif
