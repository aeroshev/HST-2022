#include "matrix.h"


Matrix::Matrix(const uint32_t size_): n_rows(size_), n_columns(size_) {
    allocSpace();
    for (uint32_t i = 0; i < n_rows; ++i) {
        for (int j = 0; j < n_columns; ++j) {
            data[i][j] = 0;
        }
    }
}

Matrix::Matrix(): n_rows(1), n_columns(1) {
    allocSpace();
    data[0][0] = 0;
}

Matrix::~Matrix() {
    for (int i = 0; i < n_rows; ++i) {
        delete[] data[i];
    }
    delete[] data;
}

double Matrix::count_under_diagonal() {
    double sum = 0.0;
    uint32_t left_side;

    for (uint32_t row = 0; row < this->n_rows; row++) {
        left_side = row + 1;
        for (uint32_t column = left_side; column < this->n_columns; column++) {
            sum += this->data[row][column];
        }
    }

    return sum;
}

std::ostream& operator<<(std::ostream& out, const Matrix &mat) {
    for (uint32_t row = 0; row < mat.n_rows; row++) {
        for (uint32_t column = 0; column < mat.n_columns; column++) {
            out << mat.data[row][column] << ' ';
        }
        out << '\n';
    }
    return out;
}

std::istream& operator>>(std::istream& in, Matrix &mat) {
    std::string ss_i;
    for (uint32_t row = 0; row < mat.n_rows; row++) {
        std::getline(in, ss_i);
        std::istringstream stringstream(ss_i);
        for (uint32_t column = 0; column < mat.n_columns; column++) {
            stringstream >> mat.data[row][column];
        }
    }
    return in;
}

size_t Matrix::size() {
    return this->n_rows * this->n_columns * sizeof(double);
}

void Matrix::allocSpace()
{
    data = new double*[n_rows];
    for (int i = 0; i < n_rows; ++i) {
        data[i] = new double[n_columns];
    }
}
