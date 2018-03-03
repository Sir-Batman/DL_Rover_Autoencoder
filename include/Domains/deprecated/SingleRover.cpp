#include "SingleRover.h"
#include <iostream>

// Constructor: Initialises physical properties of domain given size of the grid world and number of POIs. Currenly hard coded for 4 input 2 output NN control policies. Also initialises simulation properties given number of timesteps per learning epoch and toggles boolean for writing NN evaluations to file. 
SingleRover::SingleRover(matrix1d wLims, size_t nPOIs, size_t n, size_t nPop): worldLimits(wLims), numPOIs(nPOIs), nSteps(n), outputEval(false){
  InitialiseNewLearningEpoch() ;
  size_t input_size = 4 ; // hard coded for 4 element input (body frame quadrant decomposition)
  size_t output_size = 2 ; // hard coded for 2 element output [dx,dy]
  RoverNE = new NeuroEvo(input_size, output_size, 2*input_size, nPop) ;
}

// Destructor: Deletes NE control policies and closes write file
SingleRover::~SingleRover(){
  if (outputEval)
    evalFile.close() ;
  if (outputTraj){
    trajFile.close() ;
    POIFile.close() ;
  }
  delete(RoverNE) ;
  RoverNE = 0 ;
}

// Main learning execution step, includes evolution competition and resetting the environment after the first learning epoch
void SingleRover::ExecuteLearning(size_t nEpochs){
  for (size_t i = 0; i < nEpochs; i++){
    std::cout << "Episode " << i << "..." ;
    if (i > 0){
      RoverNE->EvolvePopulation(epochEvals) ;
//      epochEvals.clear() ;
      InitialiseNewLearningEpoch() ;
    }
    
    RoverNE->MutatePopulation() ;
    if (i == nEpochs-1 && outputTraj)
      SimulateEpoch(true) ;
    else
      SimulateEpoch() ;
    
    std::cout << "complete!\n" ;
  }
    
  if (outputEval)
    evalFile << "\n" ;
}

// Initial simulation parameters, includes setting initial rover position, POI positions and values, and clearing the evaluation storage vector
void SingleRover::InitialiseNewLearningEpoch(){
  // Clear initial world properties
  initialXY.clear() ;
  POIs.clear() ;
  epochEvals.clear() ;
  maxPossibleEval = 0.0 ;
  
  // Initial XY location and heading in global frame (restricted to within inner region of 9 grid)
  double rangeX = worldLimits[1] - worldLimits[0] ;
  double rangeY = worldLimits[3] - worldLimits[2] ;
  initialXY.push_back(rand(worldLimits[0]+rangeX/3.0,worldLimits[1]-rangeX/3.0)) ;
  initialXY.push_back(rand(worldLimits[2]+rangeY/3.0,worldLimits[3]-rangeX/3.0)) ;
  initialPsi = rand(-PI,PI) ;
  
  // POI locations and values in global frame (restricted to within outer regions of 9 grid)
  for (size_t i = 0; i < numPOIs; i++){
    matrix1d xy ;
    double x, y ;
    bool accept = false ;
    while (!accept){
      x = rand(worldLimits[0],worldLimits[1]) ;
      y = rand(worldLimits[0],worldLimits[1]) ;
      if (x > worldLimits[0]+rangeX/3.0 && x < worldLimits[1]-rangeX/3.0 && y > worldLimits[2]+rangeY/3.0 && y < worldLimits[3]-rangeX/3.0) {}
      else accept = true ;
    }
    xy.push_back(x) ; // x location
    xy.push_back(y) ; // y location
    double v = rand(1,10) ; // value
    POIs.push_back(Target(xy,v)) ;
    maxPossibleEval += v ; // compute maximum achievable performance
  }
}

// Simulation loop, tests each NN in the current population in the simulation world. Each simulation starts with the same configuration of rover location and POI locations and values.
void SingleRover::SimulateEpoch(bool write){
  // Write POI configuration to file
  if (write && outputTraj)
    for (size_t i = 0; i < numPOIs; i++)
      POIFile << POIs[i].GetLocation()[0] << "," << POIs[i].GetLocation()[1] << "," << POIs[i].GetValue() << "\n" ;
  
  double maxEval = 0.0 ;
  for (size_t i = 0; i < RoverNE->GetCurrentPopSize(); i++){
    matrix1d xy = initialXY ;
    double psi = initialPsi ;
    
    // Write current global state to file
    if (write)
      trajFile << xy[0] << "," << xy[1] << "," << psi << "\n" ;
    
    for (size_t t = 0; t < nSteps; t++){
      // Calculate body frame NN input state
      matrix1d s = ComputeNNInput(xy, psi) ;
      
      // Calculate body frame action
      matrix1d a = unit_vector(RoverNE->GetNNIndex(i)->EvaluateNN(s)) ;
      
      // Transform to global frame
      matrix2d Body2Global = RotationMatrix(psi) ;
      matrix1d deltaXY = matrix_mult(Body2Global,a) ;
      double deltaPsi = atan2(a[1],a[0]) ;
      
      // Move
      xy[0] += deltaXY[0] ;
      xy[1] += deltaXY[1] ;
      psi += deltaPsi ;
      psi = pi_2_pi(psi) ;
      
      // Compute observations
      for (size_t j = 0; j < numPOIs; j++)
        POIs[j].ObserveTarget(xy) ;
      
      // Write current global state to file
      if (write && outputTraj)
        trajFile << xy[0] << "," << xy[1] << "," << psi << "\n" ;
    }
    
    // Evaluate NN
    double eval = 0.0 ;
    for (size_t j = 0; j < numPOIs; j++){
      eval += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
      POIs[j].ResetTarget() ;
    }
    
    epochEvals.push_back(eval) ;
    
    if (eval > maxEval)
      maxEval = eval ;
    
  }
  if (outputEval)
    evalFile << maxEval/maxPossibleEval << "," ; // output as fraction of maximum achievable performance
}

// Wrapper for writing epoch evaluations to specified files
void SingleRover::OutputPerformance(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
  fileName << A ;
  evalFile.open(fileName.str().c_str(),std::ios::app) ;
  
  outputEval = true ;
}

// Wrapper for writing final trajectories to specified files
void SingleRover::OutputTrajectories(char * A, char * B){
	// Filename to write trajectories to stored in A
	std::stringstream tfileName ;
  tfileName << A ;
  trajFile.open(tfileName.str().c_str(),std::ios::app) ;
  
  // Filename to write POIs to stored in B
	std::stringstream pfileName ;
  pfileName << B ;
  POIFile.open(pfileName.str().c_str(),std::ios::app) ;
  
  outputTraj = true ;
}

// Compute the NN input state given the rover location and the POI locations and values in the world
matrix1d SingleRover::ComputeNNInput(matrix1d xy, double psi){
  matrix1d s = zeros(4) ;
  matrix2d Global2Body = RotationMatrix(-psi) ;
  matrix1d POIv = zeros(2) ;
  for (size_t i = 0; i < numPOIs; i++){
    POIv[0] = POIs[i].GetLocation()[0] - xy[0] ;
    POIv[1] = POIs[i].GetLocation()[1] - xy[1] ;
    matrix1d POIbody = matrix_mult(Global2Body,POIv) ;
    double d = L2_norm(xy,POIbody) ;
    double theta = atan2(POIbody[1],POIbody[0]) ;
    size_t q ;
    if (theta >= PI/2.0)
      q = 3 ;
    else if (theta >= 0.0)
      q = 0 ;
    else if (theta >= -PI/2.0)
      q = 1 ;
    else
      q = 2 ;
    s[q] += POIs[i].GetValue()/max(d,1.0) ;
//    std::cout << "Rover global state: (" << xy[0] << "," << xy[1] << "," << psi*180.0/PI 
//    << "), POI global location: (" << POIs[i].GetLocation()[0] << "," << POIs[i].GetLocation()[1] 
//    << "), POI body location: (" << POIbody[0] << "," << POIbody[1]
//    << "), bearing: " << theta*180.0/PI << ", quadrant: " << q << std::endl ;
  }
//  std::cout << "State: [" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n" ;
  return s ;
}

matrix2d SingleRover::RotationMatrix(double psi){
  matrix2d R = zeros(2,2) ;
  R[0][0] = cos(psi) ;
  R[0][1] = -sin(psi) ;
  R[1][0] = sin(psi) ;
  R[1][1] = cos(psi) ;
  return R ;
}
