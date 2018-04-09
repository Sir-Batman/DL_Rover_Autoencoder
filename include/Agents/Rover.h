#ifndef ROVER_H_
#define ROVER_H_

#include <stdlib.h>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <cmath>
#include <math.h>
#include <float.h>
#include <Eigen/Eigen>
#include "Learning/NeuroEvo.h"
#include "Utilities/Utilities.h"
#include "Target/Target.h"
//#include "POMDPs/POMDP.h"

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

using std::string ;
using std::vector ;
using std::list ;
using std::max ;
using easymath::pi_2_pi ;

class Rover{
	public:
		Rover(size_t n, size_t nPop, string evalFunc) ;
		~Rover() ;

		void computeLaserDataSimple(double distance, double theta, std::vector<double>& laserData);
		void computeLaserDataComplex(double distance, double theta, double object_radius, std::vector<double>& laserData);

		void generateLaserData(vector<Vector2d> jointState, std::ofstream& roverLaserStream, std::ofstream& poiLaserStream);

		void ResetEpochEvals() ;
		void InitialiseNewLearningEpoch(vector<Target>, Vector2d, double) ;
		void ResetStepwiseEval() ;

		Vector2d ExecuteNNControlPolicy(size_t , vector<Vector2d>) ; // executes NN_i from current (x,y,psi), outputs new (x,y)
		void ComputeStepwiseEval(vector<Vector2d>, double) ;
		void SetEpochPerformance(double G, size_t i) ;
		vector<double> GetEpochEvals(){return epochEvals ;}

		void EvolvePolicies(bool init = false) ;

		void OutputNNs(char *) ;
		NeuroEvo * GetNEPopulation(){return RoverNE ;}

		void UpdateNNStateInputCalculation(bool, size_t) ;
		bool IsStateObsUpdated(){return stateObsUpdate ;}

		double GetAverageR() ;
	protected:
		size_t nSteps ;
		size_t popSize ;
		size_t numIn ;
		size_t numOut ;
		size_t numHidden ;

		Vector2d initialXY ;
		double initialPsi ;
		vector<Target> POIs ;
		Vector2d currentXY ;
		double currentPsi ;

		bool isD ;
		double stepwiseD ;
		vector<double> epochEvals ;
		NeuroEvo * RoverNE ;
		list<double> runningAvgR ;
		size_t windowSize ;

		VectorXd ComputeNNInput(vector<Vector2d>) ;
		Matrix2d RotationMatrix(double) ;

		bool stateObsUpdate ;
		size_t goalPOI ;

		void DifferenceEvaluationFunction(vector<Vector2d>, double) ;
		void UpdatedStateEvaluationFunction(vector<Vector2d>, double) ;

		// Maximum distance of Laser Scan Data
		// sqrt(30^2 + 30^2) = 42.426 (rounded to 43)
		double LASER_DIST_MAX = 43.0;
		double POI_RADIUS = 1.0;
		double ROVER_RADIUS = 1.0;
} ;

template <class T>
void printVector(std::vector<T> v, std::ostream& stream){

    stream << v[0];

    for (size_t i = 1; i < v.size(); ++i){
        stream << ", " << v[i];
    }
    stream << "\n";
    stream << std::flush;

}

template <class T>
std::string strVector(const std::vector<T> &v)
{
    std::string s;
    s += "[";
    for (int i = 0; i < v.size()-1; ++i)
    {
        s.append(std::to_string(v[i]));
        s += ",";
    }
    s.append(std::to_string(v[v.size()-1]));
    s+="]";
    return s;
}

void send(const std::string & message, const std::string & filename)
{
	std::ofstream topy(filename);
    //std::cout << "Sending: " << message << std::endl;
	topy << message;
	topy.close();
}

std::string recieve(const std::string &filename)
{
	std::string message;
	//std::cout << "Opening file... " << std::endl;
	std::ifstream tocpp(filename);
	//std::cout << "Opened!" << std::endl;
	tocpp >> message;
	//std::cout << "Finished writing" << std::endl;
	tocpp.close();
	//std::cout << " and closed" << std::endl;
	return message;
}



#endif // ROVER_H_
