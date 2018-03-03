#include "NeuralNet.h"

// Constructor: Initialises NN given layer sizes, also initialises NN activation function, currently has hardcoded mutation rates, mutation value std and bias node value
NeuralNet::NeuralNet(size_t numIn, size_t numOut, size_t numHidden){
  bias = 1.0 ;
  weightsA = zeros(numIn, numHidden) ;
  weightsB = zeros(numHidden+1, numOut) ;
  mutationRate = 0.5 ;
  mutationStd = 1.0 ;
  
  ActivationFunction = &NeuralNet::HyperbolicTangent ;
  InitialiseWeights(weightsA) ;
  InitialiseWeights(weightsB) ;
}

// Evaluate NN output given input vector
matrix1d NeuralNet::EvaluateNN(matrix1d inputs){
  matrix1d hiddenLayer = (this->*ActivationFunction)(inputs, 0) ;
  matrix1d outputs = (this->*ActivationFunction)(hiddenLayer, 1) ;
  return outputs ;
}


// Mutate the weights of the NN according to the mutation rate and mutation value std
void NeuralNet::MutateWeights(){
  double fan_in = weightsA.size() ;
  for (size_t i = 0; i < weightsA.size(); i++)
    for (size_t j = 0; j < weightsA[0].size(); j++)
      weightsA[i][j] += RandomMutation(fan_in) ;
  
  fan_in = weightsB.size() ;
  for (size_t i = 0; i < weightsB.size(); i++)
    for (size_t j = 0; j < weightsB[0].size(); j++)
      weightsB[i][j] += RandomMutation(fan_in) ;
}

// Migrated from rebhuhnc/libraries/SingleAgent/NeuralNet/NeuralNet.cpp
double NeuralNet::RandomMutation(double fan_in) {
  // Adds random amount mutationRate% of the time,
  // amount based on fan_in and mutstd
  if (rand(0, 1) > mutationRate)
    return 0.0;
  else {
    // FOR MUTATION
    std::default_random_engine generator;
    generator.seed(static_cast<size_t>(time(NULL)));
    std::normal_distribution<double> distribution(0.0, mutationStd);
    return distribution(generator);
  }
}

// Assign weight matrices
void NeuralNet::SetWeights(matrix2d A, matrix2d B){
  weightsA = A ;
  weightsB = B ;
}

// Wrapper for writing NN weight matrices to specified files
void NeuralNet::OutputNN(const char * A, const char * B){
  // Write NN weights to txt files
  // File names stored in A and B
	std::stringstream NNFileNameA ;
	NNFileNameA << A ;
	std::stringstream NNFileNameB ;
	NNFileNameB << B ;

  WriteNN(weightsA, NNFileNameA) ;
  WriteNN(weightsB, NNFileNameB) ;
}

// Write weight matrix values to file
void NeuralNet::WriteNN(matrix2d A, std::stringstream &fileName){
  std::ofstream NNFile ;
  NNFile.open(fileName.str().c_str()) ;
  for (size_t i = 0; i < A.size(); i++){
	  for (size_t j = 0; j < A[0].size(); j++)
	    NNFile << A[i][j] << "," ;
    NNFile << "\n" ;
	}
	NNFile.close() ;
}

// Initialise NN weight matrices to random values
void NeuralNet::InitialiseWeights(matrix2d & A){
  double fan_in = A.size() ;
  for (size_t i = 0; i < A.size(); i++){
    for (size_t j = 0; j< A[0].size(); j++){
      // For initialization of the neural net weights
      double rand_neg1to1 = rand(-1, 1)*0.1;
      double scale_factor = 100.0;
      A[i][j] = scale_factor*rand_neg1to1 / sqrt(fan_in);
    }
  }
}

// Hyperbolic tan activation function
matrix1d NeuralNet::HyperbolicTangent(matrix1d input, size_t layer){
  matrix1d output ;
  if (layer == 0){
    output = matrix_mult(input, weightsA) ;
  }
  else if (layer == 1){
    input.push_back(bias) ;
    output = matrix_mult(input, weightsB) ;
  }
  else{
    std::printf("Error: second argument must be in {0,1}!\n") ;
  }
  for (size_t i = 0; i < output.size(); i++)
    output[i] = tanh(output[i]) ;
  
  return output ;
}

// Logistic function activation function
matrix1d NeuralNet::LogisticFunction(matrix1d input, size_t layer){
  matrix1d output ;
  if (layer == 0){
    output = matrix_mult(input, weightsA) ;
  }
  else if (layer == 1){
    input.push_back(bias) ;
    output = matrix_mult(input, weightsB) ;
  }
  else{
    std::printf("Error: second argument must be in {0,1}!\n") ;
  }
  for (size_t i = 0; i < output.size(); i++)
    output[i] = 1/(1+exp(-output[i])) ;
  
  return output ;
}
