#ifndef NEUR0_EVO_H_
#define NEUR0_EVO_H_

#include <chrono>
#include <algorithm>
#include <random>
#include <vector>
#include "NeuralNet.h"
#include "Utilities/MatrixTypes.h"

using std::vector ;
using std::sort ;

class NeuroEvo{
  public:
    NeuroEvo(size_t, size_t, size_t, size_t) ; // nIn, nOut, nHidden, popSize
    ~NeuroEvo() ;
    
    void MutatePopulation() ;
    void EvolvePopulation(matrix1d) ;
    matrix1d GetAllEvaluations() ;
    
    NeuralNet * GetNNIndex(size_t i){return populationNN[i] ;}
    size_t GetCurrentPopSize(){return populationNN.size() ;}
  private:
    size_t numIn ;
    size_t numOut ;
    size_t numHidden ;
    
    size_t populationSize ;
    vector<NeuralNet *> populationNN ;
    
    void (NeuroEvo::*SurvivalFunction)() ;
    void BinaryTournament() ;
    void RetainBestHalf() ;
    static bool CompareEvaluations(NeuralNet *, NeuralNet *) ;
} ;
#endif // NEUR0_EVO_H_
