// Copyright 2016 Carrie Rebhuhn
#include "MatrixTypes.h"

double easymath::normalize(double val, double min, double max) {
    return (val - min) / (max - min);
}

void easymath::zero(matrix2d * m) {
    if (m->empty())
        return;
    else
        *m = matrix2d(m->size(), matrix1d(m[0].size(), 0.0));
}

void easymath::zero(matrix1d * m) {
    *m = matrix1d(m->size(), 0.0);
}

matrix1d easymath::zeros(size_t dim1) {
    return matrix1d(dim1, 0.0);
}

matrix2d easymath::zeros(size_t dim1, size_t dim2) {
    return matrix2d(dim1, easymath::zeros(dim2));
}

matrix3d easymath::zeros(size_t dim1, size_t dim2, size_t dim3) {
    return matrix3d(dim1, easymath::zeros(dim2, dim3));
}
