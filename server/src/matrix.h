#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>


class Matrix {
    private:
        uint32_t n_rows;
        uint32_t n_columns;

        double **data;
        void allocSpace();
    public:
        Matrix();
        Matrix(const uint32_t);
        ~Matrix();

        inline double& operator()(uint32_t x, uint32_t y) { return data[x][y]; }

        friend std::ostream& operator<<(std::ostream&, const Matrix &);
        friend std::istream& operator>>(std::istream&, Matrix &);

        double count_under_diagonal();
        void read_matrix();

        size_t size();
};

#endif