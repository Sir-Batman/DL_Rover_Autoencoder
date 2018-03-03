#ifndef MAP_ELITES_H_
#define MAP_ELITES_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <float.h>
#include <math.h>
#include "Utilities/MatrixTypes.h"
#include "Learning/NeuralNet.h"

using std::vector ;
using easymath::dot_multiply ;
using easymath::sum ;

class MAPElites{
  public:
    MAPElites(matrix2d, size_t, size_t, size_t) ;
    ~MAPElites() ;
    
    double GetPerformance(matrix1d) ;
    bool IsVisited(matrix1d) ;
    NeuralNet * GetNeuralNet(matrix1d) ;
    NeuralNet * GetNeuralNet(size_t) ;
    void UpdateMap(NeuralNet *, matrix1d, double) ;
    
    size_t GetIndex(matrix1d) ;
    matrix1d GetBehaviour(size_t) ;
    
    size_t GetBDim(){return bDim ;}
    matrix1d GetPerformanceLog(){return performanceLog ;}
    vector<bool> GetFilledLog(){return behaviourFilled ;}
    
    void WriteBPMapBinary(char *) ;
    void ReadBPMapBinary(char *) ;
    
    void WritePerformanceBinary(char *) ;
    void ReadPerformanceBinary(char *) ;
    
    void WriteVisitedBinary(char *) ;
    void ReadVisitedBinary(char *) ;
  private:
    size_t bDim ;
    vector<size_t> numBins ;
    matrix2d binLimits ;
    matrix1d cProd ;
    vector<NeuralNet *> behaviourMap ;
    matrix1d performanceLog ;
    vector<bool> behaviourFilled ;
} ;
#endif // MAP_ELITE_H_
