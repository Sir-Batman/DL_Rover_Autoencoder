#ifndef DISTMULTROVER_H
#define DISTMULTROVER_H
#include "MultiRover.h"

class DistributedMR: public MultiRover {
	public:
    	void SimulateEpoch(size_t goalPOI, char * pomdpEnv, char * pomdpPolicy, VectorXd prior) ;

	protected:



};

#endif
