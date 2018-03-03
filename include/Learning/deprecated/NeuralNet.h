// Some code migrated from rebhuhnc/libraries/SingleAgent/NeuralNet/NeuralNet.h
#ifndef NEURAL_NET_H_
#define NEURAL_NET_H_

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <fstream>
#include <sstream>
#include "Utilities/UtilFunctions.h"

using easymath::rand ;
using easymath::matrix_mult ;
using easymath::zeros ;

class NeuralNet{
  public:
    NeuralNet(size_t numIn, size_t numOut, size_t numHidden) ; // single hidden layer
    ~NeuralNet(){}
    
    matrix1d EvaluateNN(matrix1d inputs) ;
    void MutateWeights() ;
    void SetWeights(matrix2d, matrix2d) ;
    matrix2d GetWeightsA() {return weightsA ;}
    matrix2d GetWeightsB() {return weightsB ;}
    void OutputNN(const char *, const char *) ; // write NN weights to file
    double GetEvaluation() {return evaluation ;}
    void SetEvaluation(double eval) {evaluation = eval ;}
  private:
    double bias ;
    matrix2d weightsA ;
    matrix2d weightsB ;
    double mutationRate ;
    double mutationStd ;
    double evaluation ;
    
    void InitialiseWeights(matrix2d &) ;
    matrix1d (NeuralNet::*ActivationFunction)(matrix1d, size_t) ;
    matrix1d HyperbolicTangent(matrix1d, size_t) ; // outputs between [-1,1]
    matrix1d LogisticFunction(matrix1d, size_t) ; // outputs between [0,1]
    double RandomMutation(double) ;
    void WriteNN(matrix2d, std::stringstream &) ;
} ;
#endif // NEURAL_NET_H_
