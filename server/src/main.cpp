#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "matrix.h"
#include "executor.h"


int main() {
    std::cout << "Run main" << '\n';
    std::string base_path("/Users/19817152/Documents/Yandex.Disk/Hybrid SupeComputers/data");
    std::string matrix_path(base_path);
    std::string results_path(base_path);
    matrix_path.append("/matrixes.dat");
    results_path.append("/results.dat");

    std::ifstream matrix_file(matrix_path.c_str());
    std::ofstream result_file(results_path.c_str());
    if (!matrix_file.is_open() || !result_file.is_open()) {
      std::cout << "Motherfucker" << '\n';
      exit(1);
    }

    Executor *executor = new Executor();
    executor->setup_from_file(matrix_file);
    executor->execute();
    executor->save_result_to_file(result_file);

    matrix_file.close();
    result_file.close();
    delete executor;

    return 0;
}
