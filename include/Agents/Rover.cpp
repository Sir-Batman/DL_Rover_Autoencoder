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


int sendcount=0;
int recvcount=0;
// Compute the NN input state given the rover locations and the POI locations and values in the world
VectorXd Rover::ComputeNNInput(vector<Vector2d> jointState){
	VectorXd s ;
	s.setZero(numIn,1) ;
	MatrixXd Global2Body = RotationMatrix(-currentPsi) ;

	std::vector< std::vector<double> > minDistancesToPOI;

	// Compute POI 360 Laser Scan Data

	// Maximum distance of Laser Scan Data
	double LASER_DIST_MAX = 43.0;
	double POI_RADIUS = 2.0;
	double ROVER_RADIUS = 2.0;
	Vector2d POIv ;

	// Initialization of POI Laser Scan
	std::vector<double> POI_Laser(360, LASER_DIST_MAX);

	// Placeholder for each POI vector POIv preinitialized above
	POIv.setZero(2,1) ;

	for (size_t i = 0; i < POIs.size(); i++){

		// Compute relative angle and distance
		// Taken from Jen Jen's code (hope it's correct)
		POIv = POIs[i].GetLocation() - currentXY ;
		Vector2d POIbody = Global2Body*POIv ;
		Vector2d diff = currentXY - POIbody ;

		// distance between the centers of our object and the POI
		double d = diff.norm() ;
		// angle between our object and POI
		double theta = atan2(POIbody(1),POIbody(0)) ;

		computeLaserDataComplex(d, theta, POI_RADIUS, POI_Laser);

	}
	// Compute Rover 360 Laser Data

	// Initialization of ROV Laser Scan
	std::vector<double> ROV_Laser(360, LASER_DIST_MAX);

	// Placeholder for each Rover vector rovV preinitialized above
	Vector2d rovV ;
	rovV.setZero(2,1) ;

	size_t ind = 0 ; // stores agent's index in the joint state
	for (size_t i = 0; i < jointState.size(); i++){
		if (i != ind){
			rovV = jointState[i] - currentXY ;
			Vector2d rovBody = Global2Body*rovV ;
			Vector2d diff = currentXY - rovBody ;
			double d = diff.norm() ;
			double theta = atan2(rovBody(1),rovBody(0)) ;

			computeLaserDataComplex(d, theta, ROVER_RADIUS, ROV_Laser);

		}
	}

	//std::cout << "\nPOI Laser Datas: ";
	//printVector(POI_Laser, std::cout);

	//std::cout << "\nROV Laser Data: ";
	//printVector(ROV_Laser, std::cout);
    //std::cout << strVector(POI_Laser) << std::endl;
    //std::cout << strVector(ROV_Laser) << std::endl;

    // Send
    std::string message;
    message += "[";
    message.append(strVector(POI_Laser));
    message += ",";
    message.append(strVector(ROV_Laser));
    message += "]";
    send(message, "./topy");
    sendcount+=1;
    //std::cout << "Sent: " << sendcount << " Recv: " << recvcount << std::endl;
    
    // Receive
    std::string response = recieve("./tocpp");
    for (int i = 0; i < 8; ++i){
        response.replace(response.find(","), 1, " ");
    }
    //std::cout << "Response: " << response << std::endl;
    std::istringstream in(response);
    for (int i = 0; i < 8; ++i)
    {
        in >> s[i];
    }
    //std::cout << s << std::endl;
    recvcount+=1;
    //std::cout << "Sent: " << sendcount << " Recv: " << recvcount << std::endl;

	return s ;
}

void Rover::generateLaserData(vector<Vector2d> jointState, std::ofstream& roverLaserStream, std::ofstream& poiLaserStream){
	// Finds the index of the current agent in the joint state?
	size_t ind = 0 ; // stores agent's index in the joint state
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
	}

	MatrixXd Global2Body = RotationMatrix(-currentPsi) ;

	// Compute POI 360 Laser Scan Data

	// Maximum distance of Laser Scan Data
	// sqrt(30^2 + 30^2) = 42.426 (rounded to 43)
	double LASER_DIST_MAX = 43.0;
	double POI_RADIUS = 1.0;
	double ROVER_RADIUS = 1.0;

	// Initialization of POI Laser Scan
	std::vector<double> POI_Laser(360, LASER_DIST_MAX);

	// Initialize POI vector
	Vector2d POIv ;
	POIv.setZero(2,1) ;

	for (size_t i = 0; i < POIs.size(); i++){

		// Compute relative angle and distance
		// Taken from Jen Jen's code (hope it's correct)
		POIv = POIs[i].GetLocation() - currentXY ;
		Vector2d POIbody = Global2Body*POIv ;
		Vector2d diff = currentXY - POIbody ;

		// distance between the centers of our object and the POI
		double d = diff.norm() ;
		// angle between our object and POI
		double theta = atan2(POIbody(1),POIbody(0)) ;

		// computeLaserDataSimple(d, theta, POI_Laser);
		computeLaserDataComplex(d, theta, POI_RADIUS, POI_Laser);

	}

	// Compute Rover 360 Laser Data

	// Initialization of ROV Laser Scan
	std::vector<double> ROV_Laser(360, LASER_DIST_MAX);

	// Initialize ROV vector
	Vector2d rovV;
	rovV.setZero(2,1);

	for (size_t i = 0; i < jointState.size(); i++){
		if (i != ind){
			rovV = jointState[i] - currentXY ;
			Vector2d rovBody = Global2Body*rovV ;
			Vector2d diff = currentXY - rovBody ;

			double d = diff.norm() ;

			double theta = atan2(rovBody(1),rovBody(0)) ;

			// computeLaserDataSimple(d, theta, ROV_Laser);
			computeLaserDataComplex(d, theta, ROVER_RADIUS, ROV_Laser);

		}
	}

	printVector(POI_Laser, poiLaserStream);

	printVector(ROV_Laser,roverLaserStream);

}

void Rover::computeLaserDataSimple(double distance, double theta, std::vector<double>& laserData){
	// Convert to Degrees and normalize angle to range [0, 360] rather than [-180, 180]
	int theta_deg = std::round((theta * 180 / PI) + 360);


	// If no closer object has been detected, update the laser scan data
	if (distance < laserData[theta_deg % 360]){
		laserData[theta_deg % 360] = distance;
	}
}

void Rover::computeLaserDataComplex(double distance, double angle_to_object, double object_radius, std::vector<double>& laserData){

	// the angle from angle_to_object 
	// from which the object will return laser information (in radians)
	// (a) in trig.jpg
	double angles_covered = std::atan(object_radius/distance);

	// Convert to Degrees and normalize angle to range [0, 360] rather than [-180, 180]
	double angle_to_object_deg = (angle_to_object * 180 / PI) + 360;		
	double angles_covered_deg = angles_covered * 180 / PI;


	// These variables match with the math in trig.jpg
	// We use these variables to determine what the laser return should be.
	// (b) in trig.jpg
	double alpha = angles_covered_deg;
	double d = distance;
	double r = object_radius;

	double gamma = 180 - 90 - alpha - 45;
	double beta = 180 - alpha - gamma;


	double lambda = 0;
	double dist = 0;


	for (int theta = 0; theta <= alpha; ++theta)
	{
		// Assuming roughly round object, compute the laser return signal
		// See (b) in trig.jpg
		lambda = 180 - theta - beta;
		dist = std::sin(beta)*(d-r) / sin(lambda);

		// If no closer object has been detected, update the laser scan data
		if (dist < laserData[int(angle_to_object_deg + theta) % 360]){
			laserData[int(angle_to_object_deg + theta) % 360] = dist;
		}
		if (dist < laserData[int(angle_to_object_deg - theta) % 360]){
			laserData[int(angle_to_object_deg - theta) % 360] = dist;
		}
	}


	// int angle = std::round(theta_deg);
	// if (distance < laserData[angle % 360]){
	// 	glaserData[angle % 360] = distance;
	// }
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
