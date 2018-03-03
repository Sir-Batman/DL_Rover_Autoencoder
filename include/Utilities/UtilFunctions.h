// Some functions migrated from rebhuhnc/libraries/Math/easymath.h
#ifndef UTIL_FUNCTIONS_H_
#define UTIL_FUNCTIONS_H_

#ifndef PI
#define PI 3.14159265358979323846264338328
#endif

#include "MatrixTypes.h"
#include "XY.h"
#include <math.h>

namespace easymath {
// Returns L2 norm of two vectors
double L2_norm(matrix1d a, matrix1d b) ;
double L2_norm(XY a, XY b) ;

// Returns dot product of two vectors
matrix1d dot_multiply(matrix1d a) ;
matrix1d dot_multiply(matrix1d a, matrix1d b) ;

// Returns matrix2d multiplication
matrix2d matrix_mult(matrix2d A, matrix2d B) ;
matrix1d matrix_mult(matrix1d A, matrix2d B) ;
matrix1d matrix_mult(matrix2d A, matrix1d B) ;

// Returns determinant of square matrix
double determinant(matrix2d) ;

// Returns Cholesky decomposition (lower triangular matrix)
matrix2d chol(matrix2d) ;

// Returns matrix inverse
matrix2d inverse(matrix2d) ;

// Returns the unit vector
matrix1d unit_vector(matrix1d) ;

// Returns a random number between two values
double rand(double low, double high) ;

// Normalise angles between +/-PI
double pi_2_pi(double) ;
} // namespace easymath
#endif // UTIL_FUNCTIONS_H_
