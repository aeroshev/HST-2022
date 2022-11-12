#include <iostream>
#include <fstream>
#include <tuple>
#include <string>
#include <vector>
#include <sstream>
#include "diagonalMatrix.h"

using namespace std;

#define MPI_CHECK(call)                          \
  if ((call) != MPI_SUCCESS) {                   \
    cerr << "MPI error calling \"" #call "\"\n"; \
    my_abort(-1);                                \
  }

// Shut down MPI cleanly if something goes wrong
void my_abort(int err) {
  cout << "Test FAILED\n";
  MPI_Abort(MPI_COMM_WORLD, err);
}

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


int main(int argc, char **argv) {

    string base_path("/root/eroshev/HST-2022/data");

    // Intialize MPI
    ifstream matrix_file;
    ofstream result_file;
    int size, quantity, receive, receive_elems;
    vector<float> dataRoot;
    vector<float> resultsRoot;

    int numtasks, rank;
    MPI_CHECK(MPI_Init(&argc, &argv));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numtasks));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    if (rank == 0) {
        cout << "Root process. Initalize parameters" << '\n';
        cout << "Num procs: " << numtasks << '\n';

        string matrix_path(base_path);
        string results_path(base_path);
        matrix_path.append("/small_matrixes.dat");
        results_path.append("/small_results.dat");

        matrix_file.open(matrix_path.c_str());
        result_file.open(results_path.c_str());
        if (!matrix_file.is_open() || !result_file.is_open()) {
            cout << "Files not founds" << '\n';
            exit(1);
        }

        cout << "Read matrix" << "\n";
        tie(size, quantity) = read_matrix(matrix_file, dataRoot);
        resultsRoot.resize(quantity);

        receive = quantity / numtasks;
        receive_elems = size * size * receive;

        cout << "Size: " << size << "\n";
        cout << "Receive: " << receive << "\n";
        cout << "Receive elems: " << receive_elems << "\n";
    }
    /*
    Sending shared variables from root
    */
    MPI_CHECK(MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&receive, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Bcast(&receive_elems, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // Slave params
    vector<float> dataPartial;
    vector<float> resultsPartial;
    dataPartial.resize(receive_elems);
    resultsPartial.resize(receive);

    MPI_CHECK(MPI_Scatter(dataRoot.data(), receive_elems, MPI_FLOAT, dataPartial.data(), receive_elems, MPI_FLOAT, 0, MPI_COMM_WORLD));

    double start = MPI_Wtime();
    // Execute block
    for (size_t pos = 0; pos < receive; pos++) {
        
        resultsPartial[pos] = computeGPU(dataPartial, size, pos);
    }
    double end = MPI_Wtime();

    MPI_CHECK(MPI_Gather(resultsPartial.data(), receive, MPI_FLOAT, resultsRoot.data(), receive, MPI_FLOAT, 0, MPI_COMM_WORLD));

    double proc_time = end - start;
    double elapsed_time;

    MPI_CHECK(MPI_Reduce(&proc_time, &elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        for (auto const& result: resultsRoot) {
            result_file << result << "\n";
        }

        ostringstream string_stream;
        string_stream << "Took: " << elapsed_time << " s" << '\n';
        string_stream << "Memory: " << quantity * size * size * sizeof(float) << " bytes" << '\n';
        std::cout << string_stream.str();
        result_file << string_stream.str();

        matrix_file.close();
        result_file.close();
    }

    MPI_CHECK(MPI_Finalize());

    return 0;
}
