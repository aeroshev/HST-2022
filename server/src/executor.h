#ifndef EXECUTOR_H
#define EXECUTOR_H

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <unistd.h>
#include "matrix.h"

using namespace std::chrono;


class Executor {
    private:
        std::vector<Matrix *> pool;
        std::vector<double> results;

        duration<double, std::milli> duration;
        uint64_t memory;
    public:
        Executor();
        Executor(size_t);
        ~Executor();

        void setup_from_file(std::ifstream&);
        void execute();
        void save_result_to_file(std::ofstream&);
        
        static void c_execute(double**, double*, size_t, size_t);
};

#endif