#include "Rover.h"

Rover::Rover(size_t n, size_t nPop, string evalFunc): nSteps(n), popSize(nPop){
	numIn = 8 ; // hard coded for 4 element input (body frame quadrant decomposition)
	numOut = 2 ; // hard coded for 2 element output [dx,dy]
	numHidden = 16 ;
	RoverNE = new NeuroEvo(numIn, numOut, numHidden, nPop) ;

	if (evalFunc.compare("D") == 0)
		isD = true ;
	else if (evalFunc.compare("G") == 0)
		isD = false ;
	else{
		std::cout << "ERROR: Unknown evaluation function type [" << evalFunc << "], setting to global evaluation!\n" ;
		isD = false ;
	}
	windowSize = nSteps/10 ; // hardcoded running average window size to be 1/10 of full experimental run
	rThreshold.push_back(0.01) ;
	rThreshold.push_back(0.3) ; // hardcoded reward threshold, D logs for 1000 step executions suggest this is a good value
	pomdpAction = 0 ; // initial action is always to not ask for help
	stateObsUpdate = false ; // true if human assistance has redefined NN control policy state calculation
}

Rover::~Rover(){
	delete(RoverNE) ;
	RoverNE = 0 ;
}

void Rover::ResetEpochEvals(){
	// Re-initialise size of evaluations vector
	vector<double> evals(2*popSize,0) ;
	epochEvals = evals ;
}

// Initial simulation parameters, includes setting initial rover position, POI positions and values, and clearing the evaluation storage vector
void Rover::InitialiseNewLearningEpoch(vector<Target> pois, Vector2d xy, double psi){
	// Clear initial world properties
	initialXY.setZero(initialXY.size(),1) ;
	POIs.clear() ;

	ResetStepwiseEval() ;

	for (size_t i = 0; i < pois.size(); i++){
		POIs.push_back(pois[i]) ;
		//    maxPossibleEval += POIs[i].GetValue() ;
	}

	initialXY(0) = xy(0) ;
	initialXY(1) = xy(1) ;
	initialPsi = psi ;

	currentXY = initialXY ;
	currentPsi = initialPsi ;

	// Reinitialise expertise POMDP properties
	pomdpAction = 0 ;
	stateObsUpdate = false ;
}

void Rover::ResetStepwiseEval(){
	stepwiseD = 0.0 ;
	runningAvgR.clear() ;
}

Vector2d Rover::ExecuteNNControlPolicy(size_t i, vector<Vector2d> jointState){
	// Calculate body frame NN input state
	VectorXd s ;
	if (!stateObsUpdate)
		s = ComputeNNInput(jointState) ;
	else{
		vector<Vector2d> tempState ;
		tempState.clear() ;
		s = ComputeNNInput(tempState) ;
	}

	// Calculate body frame action
	VectorXd a = RoverNE->GetNNIndex(i)->EvaluateNN(s).normalized() ;

	// Transform to global frame
	Matrix2d Body2Global = RotationMatrix(currentPsi) ;
	Vector2d deltaXY = Body2Global*a ;
	double deltaPsi = atan2(a(1),a(0)) ;

	// Move
	currentXY += deltaXY ;
	currentPsi += deltaPsi ;
	currentPsi = pi_2_pi(currentPsi) ;

	return currentXY ;
}

void Rover::ComputeStepwiseEval(vector<Vector2d> jointState, double G){
	if (!stateObsUpdate)
		DifferenceEvaluationFunction(jointState, G) ;
	else
		UpdatedStateEvaluationFunction(jointState, G) ;
}

void Rover::SetEpochPerformance(double G, size_t i){
	if (isD)
		epochEvals[i] = stepwiseD ;
	else
		epochEvals[i] = G ;
}

void Rover::EvolvePolicies(bool init){
	if (!init)
		RoverNE->EvolvePopulation(epochEvals) ;
	RoverNE->MutatePopulation() ;
}

void Rover::OutputNNs(char * A){
	// Filename to write to stored in A
	std::stringstream fileName ;
	fileName << A ;
	std::ofstream NNFile ;
	NNFile.open(fileName.str().c_str(),std::ios::app) ;

	// Only write in non-mutated (competitive) policies
	for (size_t i = 0; i < popSize; i++){
		NeuralNet * NN = RoverNE->GetNNIndex(i) ;
		MatrixXd NNA = NN->GetWeightsA() ;
		for (int j = 0; j < NNA.rows(); j++){
			for (int k = 0; k < NNA.cols(); k++)
				NNFile << NNA(j,k) << "," ;
			NNFile << "\n" ;
		}

		MatrixXd NNB = NN->GetWeightsB() ;
		for (int j = 0; j < NNB.rows(); j++){
			for (int k = 0; k < NNB.cols(); k++)
				NNFile << NNB(j,k) << "," ;
			NNFile << "\n" ;
		}
	}
	NNFile.close() ;
}

void Rover::SetPOMDPPolicy(POMDP * pomdp){
	expertisePOMDP = pomdp ;
	belief = expertisePOMDP->GetBelief() ;
}

size_t Rover::ComputePOMDPAction(){
	// Wait for sufficient reward observations
	if (runningAvgR.size() == windowSize){
		// Calculate reward observation from running average
		double avgSum = GetAverageR() ;

		// Convert to discrete observation for POMDP interface
		size_t obs ;
		if (avgSum <= rThreshold[0])
			obs = 0 ;
		else if (avgSum < rThreshold[1])
			obs = 1 ;
		else
			obs = 2 ;

		// Updated POMDP with latest observation
		expertisePOMDP->UpdateBelief(pomdpAction, obs) ;
		belief = expertisePOMDP->GetBelief() ;

		// Compute next action
		pomdpAction = expertisePOMDP->GetBestAction() ;
	}
	return pomdpAction ;
}

double Rover::GetAverageR(){
	// Calculate reward observation from running average
	double avgSum = 0.0 ;
	for (list<double>::iterator it=runningAvgR.begin(); it!=runningAvgR.end(); ++it)
		avgSum += *it ;

	avgSum /= windowSize ;
	return avgSum ;
}

void Rover::UpdateNNStateInputCalculation(bool update, size_t gID){
	stateObsUpdate = update ;
	goalPOI = gID ;
	vector<Target> newPOIs ;
	newPOIs.push_back(POIs[gID]) ;
	POIs.clear() ;
	POIs.push_back(newPOIs[0]) ; // remove all other POIs from consideration in the state
	runningAvgR.clear() ; // restart running average calculation window
	pomdpAction = 0 ; // reset pomdp action
}

double MinMaxDistort(double d1, double d2, double dist)
{
	double top=0, bottom=0;
	double result = 0;

	if (dist == 0)
	{
		result = 1;
	}
	else
	{
		std::vector<double> values;
		values.push_back(d1);
		values.push_back(d2);
		values.push_back(dist);
		std::sort(values.begin(), values.end());
		d1 = values[0];
		d2 = values[1];
		top = std::max( std::min(d2/dist, 1.0) , .5);
		bottom = std::max( std::min(d1/dist, 1.0) , .5);
		result = top/bottom;
	}
	return result;
}

double GaussDistort(double d1, double d2, double dist)
{
	double result = 0;
	int n = 5;
	result = (n/(d1+d2)) * exp( -(pow(dist-d1,2)/n + (pow(dist-d2,2)/n)));
	return result;
}
	

// Compute the NN input state given the rover locations and the POI locations and values in the world
VectorXd Rover::ComputeNNInput(vector<Vector2d> jointState){
	VectorXd s ;
	s.setZero(numIn,1) ;
	MatrixXd Global2Body = RotationMatrix(-currentPsi) ;

	std::vector< std::vector<double> > minDistancesToPOI;

	// Find the minimum distances to a POI
	//For each POI
	for (size_t i = 0; i < POIs.size(); i++)
	{
		minDistancesToPOI.resize(POIs.size());
		for (size_t j = 0; j < jointState.size(); j++)
		{
			Vector2d P2Av = POIs[i].GetLocation() - jointState[j];
			double dist = P2Av.norm();
			minDistancesToPOI[i].push_back(dist);
		}
		std::sort(minDistancesToPOI[i].begin(), minDistancesToPOI[i].end());
		double myDist = 0;
		Vector2d POIvec = POIs[i].GetLocation() - currentXY ;
		Vector2d POIbody = Global2Body*POIvec ;
		Vector2d diff = currentXY - POIbody ;
		myDist = diff.norm() ;
		// Hackily store "my" distance to POI i on the back of minDistancesToPOI[i]
		minDistancesToPOI[i].push_back(myDist);
	}

	// Compute POI observation states
	Vector2d POIv ;
	POIv.setZero(2,1) ;
	for (size_t i = 0; i < POIs.size(); i++){
		POIv = POIs[i].GetLocation() - currentXY ;
		Vector2d POIbody = Global2Body*POIv ;
		Vector2d diff = currentXY - POIbody ;
		double d = diff.norm() ;
		double theta = atan2(POIbody(1),POIbody(0)) ;
		size_t q ;
		if (theta >= PI/2.0)
			q = 3 ;
		else if (theta >= 0.0)
			q = 0 ;
		else if (theta >= -PI/2.0)
			q = 1 ;
		else
			q = 2 ;

		/*
		double ScalingV = MinMaxDistort(
				minDistancesToPOI[i][0],
				minDistancesToPOI[i][1],
				minDistancesToPOI[i].back());
		*/
		/*
		double ScalingV = GaussDistort(
				minDistancesToPOI[i][0],
				minDistancesToPOI[i][1],
				minDistancesToPOI[i].back());
		*/
		double ScalingV = 1;

		s(q) += ScalingV*POIs[i].GetValue()/max(d,1.0) ;
	}
	//std::cout << "State: [" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n" ;

	// Compute rover observation states
	// Finds the index of the current agent in the joint state?
	size_t ind = 0 ; // stores agent's index in the joint state
	double minDiff = DBL_MAX ;
	for (size_t i = 0; i < jointState.size(); i++){
		if (jointState[i](0) == currentXY(0) && jointState[i](1) == currentXY(1))
		{
			ind = i;
			break;
		}
		if (i+1 == jointState.size())
		{
			std::cerr << "ERROR [Agent/Rover.cpp] Agent did not find self in joint state." << std::endl;
			exit(1);
		}
		//double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
		//if (diff < minDiff){
			//minDiff = diff ;
			//ind = i ;
		//}
	}

	Vector2d rovV ;
	rovV.setZero(2,1) ;
	for (size_t i = 0; i < jointState.size(); i++){
		if (i != ind){
			rovV = jointState[i] - currentXY ;
			Vector2d rovBody = Global2Body*rovV ;
			Vector2d diff = currentXY - rovBody ;
			double d = diff.norm() ;
			double theta = atan2(rovBody(1),rovBody(0)) ;
			size_t q ;
			if (theta >= PI/2.0)
				q = 7 ;
			else if (theta >= 0.0)
				q = 4 ;
			else if (theta >= -PI/2.0)
				q = 5 ;
			else
				q = 6 ;
			s(q) += 1.0/max(d,1.0) ;
		}
	}
	//std::cout << "Final State:\n" << s << std::endl;

	return s ;
}

Matrix2d Rover::RotationMatrix(double psi){
	Matrix2d R ;
	R(0,0) = cos(psi) ;
	R(0,1) = -sin(psi) ;
	R(1,0) = sin(psi) ;
	R(1,1) = cos(psi) ;
	return R ;
}

void Rover::DifferenceEvaluationFunction(vector<Vector2d> jointState, double G){
	double G_hat = 0 ;
	size_t ind = 0 ; // stores agent's index in the joint state
	double minDiff = DBL_MAX ;
	for (size_t i = 0; i < jointState.size(); i++){
		double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
		if (diff < minDiff){
			minDiff = diff ;
			ind = i ;
		}
	}

	// Replace agent state with counterfactual
	jointState[ind](0) = initialXY(0) ;
	jointState[ind](1) = initialXY(1) ;
	for (size_t i = 0; i < jointState.size(); i++)
		for (size_t j = 0; j < POIs.size(); j++)
			POIs[j].ObserveTarget(jointState[i]) ;

	for (size_t j = 0; j < POIs.size(); j++){
		G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
		POIs[j].ResetTarget() ;
	}

	stepwiseD += (G-G_hat) ;
	if (runningAvgR.size() == windowSize)
		runningAvgR.pop_front() ;
	runningAvgR.push_back(G) ;
}

void Rover::UpdatedStateEvaluationFunction(vector<Vector2d> jointState, double G){
	double G_hat = 0 ;
	size_t ind = 0 ; // stores agent's index in the joint state
	double minDiff = DBL_MAX ;
	for (size_t i = 0; i < jointState.size(); i++){
		double diff = sqrt(pow(jointState[i](0)-currentXY(0),2)+pow(jointState[i](1)-currentXY(1),2)) ;
		if (diff < minDiff){
			minDiff = diff ;
			ind = i ;
		}
	}
	vector<Vector2d> newState ; // ignore effect of all other agents in state
	newState.push_back(jointState[ind]) ;

	// Replace agent state with counterfactual
	newState[0](0) = initialXY(0) ;
	newState[0](1) = initialXY(1) ;
	for (size_t i = 0; i < newState.size(); i++)
		for (size_t j = 0; j < POIs.size(); j++)
			POIs[j].ObserveTarget(newState[i]) ;

	for (size_t j = 0; j < POIs.size(); j++){
		G_hat += POIs[j].IsObserved() ? (POIs[j].GetValue()/max(POIs[j].GetNearestObs(),1.0)) : 0.0 ;
		POIs[j].ResetTarget() ;
	}

	stepwiseD += (G-G_hat) ;
	if (runningAvgR.size() == windowSize)
		runningAvgR.pop_front() ;
	runningAvgR.push_back(G) ;
}
