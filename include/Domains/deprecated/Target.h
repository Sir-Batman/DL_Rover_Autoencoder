#ifndef TARGET_H_
#define TARGET_H_

#include <float.h>
#include "Utilities/MatrixTypes.h"
#include "Utilities/UtilFunctions.h"

using easymath::L2_norm ;

class Target{
  public:
    Target(matrix1d xy, double v): loc(xy), val(v), obsRadius(4.0), nearestObs(DBL_MAX), observed(false){}
    ~Target(){}
    
    matrix1d GetLocation(){return loc ;}
    double GetValue(){return val ;}
    double GetNearestObs(){return nearestObs ;}
    bool IsObserved(){return observed ;}
    
    void ObserveTarget(matrix1d xy){
      double d = L2_norm(xy,loc) ;
      if (observed && d < nearestObs)
        nearestObs = d ;
      else if (!observed && d <= obsRadius){
        nearestObs = d ;
        observed = true ;
      }
    }
    
    void ResetTarget(){
      nearestObs = DBL_MAX ;
      observed = false ;
    }
  private:
    matrix1d loc ;
    double val ;
    double obsRadius ;
    double nearestObs ;
    bool observed ;
} ;
#endif // TARGET_H_
