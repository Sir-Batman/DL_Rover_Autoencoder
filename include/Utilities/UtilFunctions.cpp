#include "UtilFunctions.h"
#include <iostream>
#include <math.h>
#include <stdlib.h>


namespace easymath{
double L2_norm(matrix1d a, matrix1d b){
  try{
    if (a.size() != b.size())
      throw "Cannot compute L2 norm, vectors of different size!" ;
    
    matrix1d c = dot_multiply(a-b) ;
    
    return sqrt(sum(c)) ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

double L2_norm(XY a, XY b){
  return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y)) ;
}

matrix1d dot_multiply(matrix1d a){
  matrix1d c ;
  for (size_t i = 0; i < a.size(); i++)
    c.push_back(a[i]*a[i]) ;
  return c ;
}

matrix1d dot_multiply(matrix1d a, matrix1d b){
  try{
    if (a.size() != b.size())
      throw "Cannot compute dot product, vectors of different size!" ;
    
    matrix1d c ;
    for (size_t i = 0; i < a.size(); i++)
      c.push_back(a[i]*b[i]) ;
    return c ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

matrix2d matrix_mult(matrix2d A, matrix2d B){
  try{
    if (A[0].size() != B.size())
      throw "ERROR [mxn]: Cannot compute matrix multiplication, matrix dimensions do not match!" ;
    
    matrix2d C = zeros(A.size(),B[0].size()) ;
    for (size_t i = 0; i < C.size(); i++){
      for (size_t j = 0; j < C[0].size(); j++){
        matrix1d b ;
        for (size_t k = 0; k < B.size(); k++)
          b.push_back(B[k][j]) ;
        C[i][j] = sum(dot_multiply(A[i],b)) ;
      }
    }
    return C ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

matrix1d matrix_mult(matrix1d A, matrix2d B){
  try{
    if (A.size() != B.size())
      throw "ERROR [1xn]: Cannot compute matrix multiplication, matrix dimensions do not match!" ;
    
    matrix1d C = zeros(B[0].size()) ;
    for (size_t j = 0; j < C.size(); j++){
      matrix1d b ;
      for (size_t k = 0; k < B.size(); k++)
        b.push_back(B[k][j]) ;
      C[j] = sum(dot_multiply(A,b)) ;
    }
    return C ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
    std::cout << A.size() << " " << B.size() << std::endl ;
  }
}

matrix1d matrix_mult(matrix2d A, matrix1d B){
  try{
    if (A[0].size() != B.size())
      throw "ERROR [nx1]: Cannot compute matrix multiplication, matrix dimensions do not match!" ;
    
    matrix1d C = zeros(A.size()) ;
    for (size_t i = 0; i < C.size(); i++)
      C[i] = sum(dot_multiply(A[i],B)) ;
    
    return C ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

double determinant(matrix2d A){
  try{
    if (A.size() != A[0].size())
      throw "ERROR [matrix determinant]: Cannot compute matrix determinant of non-square matrix!" ;
    
    // Calculate determinant
    double det ;
    return det ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

matrix2d chol(matrix2d A){
  try{
    if (A.size() != A[0].size())
      throw "ERROR [Cholesky decomposition]: Cannot compute Cholesky decomposition of non-square matrix!" ;
    
    // Calculate Cholesky decomposition
    matrix2d cholesky ;
    return cholesky ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

matrix2d inverse(matrix2d A){
  try{
    if (A.size() != A[0].size())
      throw "ERROR [Matrix inverse]: Cannot compute inverse of non-square matrix!" ;
    
    // Calculate Cholesky decomposition
    matrix2d inv ;
    return inv ;
  }
  catch(const char * msg){
    std::cerr << msg << std::endl ;
  }
}

matrix1d unit_vector(matrix1d v){
  matrix1d origin = zeros(2) ;
  double norm = L2_norm(v,origin) ;
  matrix1d u ;
  u.push_back(v[0]/norm) ;
  u.push_back(v[1]/norm) ;
  return u ;
}

double rand(double low, double high) {
    double r = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
    return r*(high - low) + low;
}

double pi_2_pi(double x){
  x = fmod(x+PI,2.0*PI) ;
  if (x < 0.0)
    x += 2.0*PI ;
  return x - PI ;
}
} // namespace easymath
