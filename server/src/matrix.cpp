#include "matrix.h"


Matrix::Matrix(const size_t size_): m_size(size_) {
    allocSpace();
    for (size_t i = 0; i < m_size; ++i) {
        for (int j = 0; j < m_size; ++j) {
            data[i * m_size + j] = 0;
        }
    }
}

Matrix::Matrix(): m_size(1) {
    allocSpace();
    data[0] = 0;
}

Matrix::~Matrix() {
    delete[] data;
}

double Matrix::count_under_diagonal() {
    double sum = 0.0;
    size_t left_side;

    for (size_t row = 0; row < this->m_size; row++) {
        left_side = row + 1;
        for (size_t column = left_side; column < this->m_size; column++) {
            sum += this->data[row * m_size + column];
        }
    }
    return sum;
}

float Matrix::c_count_under_diagonal(std::vector<float>& line_matrix, int N, int pos) {
    float sum = 0.0;
    int shift = pos * N * N;
    int left_side;

    for (int row = 0; row < N; row++) {
        left_side = row + 1;
        for (int column = left_side; column < N; column++) {
            // std::cout << "Map " << (row * N + column) + pos << "\n";
            sum += line_matrix[(row * N + column) + pos];
        }
    }

    return sum;
}

double *Matrix::elements() {
    return this->data;
}

std::ostream& operator<<(std::ostream& out, const Matrix &mat) {
    for (uint32_t row = 0; row < mat.m_size; row++) {
        for (uint32_t column = 0; column < mat.m_size; column++) {
            out << mat.data[row * mat.m_size + column] << ' ';
        }
        out << '\n';
    }
    return out;
}

std::istream& operator>>(std::istream& in, Matrix &mat) {
    std::string ss_i;
    for (uint32_t row = 0; row < mat.m_size; row++) {
        std::getline(in, ss_i);
        std::istringstream stringstream(ss_i);
        for (uint32_t column = 0; column < mat.m_size; column++) {
            stringstream >> mat.data[row * mat.m_size + column];
        }
    }
    return in;
}

size_t Matrix::size() {
    return this->m_size * this->m_size * sizeof(double);
}

void Matrix::allocSpace() {
    data = new double[m_size * m_size];
}
