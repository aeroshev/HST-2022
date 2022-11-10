#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>


class Matrix {
    private:
        size_t m_size;

        double *data;
        void allocSpace();
    public:
        Matrix();
        Matrix(const size_t);
        ~Matrix();

        inline double& operator()(size_t x, size_t y) { return data[x * m_size + y]; }

        friend std::ostream& operator<<(std::ostream&, const Matrix &);
        friend std::istream& operator>>(std::istream&, Matrix &);

        double count_under_diagonal();
        void read_matrix();
        double* elements();

        static float c_count_under_diagonal(std::vector<float>&, int, int);

        size_t size();
};

#endif