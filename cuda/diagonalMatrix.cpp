#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include "diagonalMatrix.h"

using namespace std::chrono;


tuple<int, int> read_matrix(ifstream& matrix_file, vector<float>& vector_matrices) {
    /*
    Read matrices from file when first line is size and qunatity
    after is num of matrices
    Read in one line vector of floats
    */
    int size, quantity;
    matrix_file >> size >> quantity;
    matrix_file.ignore();
    cout << "Size matrix: " << size << '\n';
    cout << "Quantity matrix: " << quantity << '\n';
    vector_matrices.resize(quantity * size * size);

    string ss_i;
    int shift = 0;
    for (int pos = 0; pos < vector_matrices.size(); pos++) {
        shift = pos * size * size;
        for (int row = 0; row < size; row++) {
            getline(matrix_file, ss_i);
            istringstream stringstream(ss_i);
            for (int column = 0; column < size; column++) {
                stringstream >> vector_matrices[(row * size + column) + shift];
            }
        }
    }
    return make_tuple(size, quantity);
}


int main(int argc, char *argv[]) {
    cout << "Run main" << '\n';
    string base_path("/root/eroshev/HST-2022/data");
    string matrix_path(base_path);
    string results_path(base_path);
    matrix_path.append("/small_matrixes.dat");
    results_path.append("/small_results.dat");

    ifstream matrix_file(matrix_path.c_str());
    ofstream result_file(results_path.c_str());
    if (!matrix_file.is_open() || !result_file.is_open()) {
        cout << "Files not found" << '\n';
        exit(1);
    }

    int size, quantity;
    vector<float> vectorMatrices, vectorResults;
    cout << "Read matrix" << "\n";
    tie(size, quantity) = read_matrix(matrix_file, vectorMatrices);
    vectorResults.resize(quantity);
    
    auto start = high_resolution_clock::now();

    for (size_t pos = 0; pos < quantity; pos++) {
        vectorResults[pos] = computeGPU(vectorMatrices, size, pos);
    }

    auto stop = high_resolution_clock::now();

    auto elapsed_time = duration_cast<seconds>(stop - start).count();

    for (auto const& result: vectorResults) {
        result_file << result << "\n";
    }

    ostringstream string_stream;
    string_stream << "Took: " << elapsed_time << " s" << '\n';
    string_stream << "Memory: " << quantity * size * size * sizeof(float) << " bytes" << '\n';
    cout << string_stream.str();
    result_file << string_stream.str();

    matrix_file.close();
    result_file.close();

    return 0;
}