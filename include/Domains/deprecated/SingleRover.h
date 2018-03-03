#ifndef SINGLE_ROVER_H_
#define SINGLE_ROVER_H_

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <math.h>
#include "Learning/NeuroEvo.h"
#include "Utilities/MatrixTypes.h"
#include "Utilities/UtilFunctions.h"
#include "Target.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using easymath::rand ;
using easymath::unit_vector ;
using easymath::L2_norm ;
using easymath::pi_2_pi ;
using std::max ;

class SingleRover{
  public:
    SingleRover(matrix1d, size_t, size_t, size_t) ;
    ~SingleRover() ;
    
    void ExecuteLearning(size_t) ;
    
    void InitialiseNewLearningEpoch() ;
    void SimulateEpoch(bool write=false) ;
    
    void OutputPerformance(char *) ; // write epoch evaluations to file
    void OutputTrajectories(char *, char *) ; // write final trajectories and POIs to file
  private:
    matrix1d worldLimits ;
    size_t numPOIs ;
    size_t nSteps ;
    
    vector<Target> POIs ;
    matrix1d initialXY ;
    double initialPsi ;
    matrix1d action ;
    
    double maxPossibleEval ;
    matrix1d epochEvals ;
    NeuroEvo * RoverNE ;
    
    bool outputEval ;
    std::ofstream evalFile ;
    
    bool outputTraj ;
    std::ofstream trajFile ;
    std::ofstream POIFile ;
    
    matrix1d ComputeNNInput(matrix1d, double) ;
    matrix2d RotationMatrix(double) ;
} ;
#endif // SINGLE_ROVER_H_
