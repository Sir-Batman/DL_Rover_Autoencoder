#ifndef MULTIROVER_H_
#define MULTIROVER_H_

#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Eigen>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>

#include "Agents/Rover.h"
//#include "POMDPs/POMDP.h"
#include "Target/Target.h"

using std::string ;
using std::vector ;
using std::shuffle ;
using namespace Eigen ;

class MultiRover{
	public:
		MultiRover(vector<double>, size_t, size_t, size_t, string, size_t, int c = 1) ;
		~MultiRover() ;

		void InitialiseEpoch() ;

		void SimulateEpoch(bool train = true) ;
		void SimulateEpoch(size_t goalPOI, char * pomdpEnv, char * pomdpPolicy, VectorXd prior) ;
		void EvolvePolicies(bool init = false) ;
		void ResetEpochEvals() ;

		void OutputPerformance(char *) ;
		void OutputTrajectories(char *, char *) ;
		void OutputControlPolicies(char *) ;
		void OutputQueries(char *) ;
		void OutputBeliefs(char *) ;
		void OutputAverageStepwise(char *) ;

		void WriteLaserData();
		void OutputLaserData(std::string poi_laser_fname, std::string rov_laser_fname);

		void ExecutePolicies(char * readFile, char * storeTraj, char * storePOI, char * storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure


		void ExecutePolicies(char * expFile, char * novFile, char * storeTraj, char * storePOI, char* storeEval, size_t numIn, size_t numOut, size_t numHidden) ; // read in expert and novice control policies and execute in random world, store trajectory and POI results in second and third inputs, team performance stored in fourth input, fifth-seventh inputs define NN structure
	protected:
		vector<double> world ;
		size_t nSteps ;
		size_t nPop ;
		size_t nPOIs ;
		string evaluationFunction ;
		size_t nRovers ;
		int coupling ;

		vector<Vector2d> initialXYs ;
		vector<double> initialPsis ;

		vector<Rover *> roverTeam ;
		vector<Target> POIs ;
		bool gPOIObs ;

		bool outputEvals ;
		bool outputTrajs ;
		bool outputNNs ;
		bool outputQury ;
		bool outputBlf ;
		bool outputAvgStepR ;

		std::ofstream evalFile ;
		std::ofstream trajFile ;
		std::ofstream POIFile ;
		std::ofstream NNFile ;
		std::ofstream quryFile ;
		std::ofstream blfFile ;
		std::ofstream avgStepRFile ;

		// additional ofstream to output laser data
		bool outputLaser;
		std::ofstream poi_laser_stream;
		std::ofstream rov_laser_stream;

		vector< vector<size_t> > RandomiseTeams(size_t) ;
} ;

#endif // MULTIROVER_H_
