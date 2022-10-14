#include "executor.h"


Executor::Executor() {};

Executor::Executor(size_t size) {
    pool.reserve(size);
    pool.reserve(size);

    memory = 0;
}

Executor::~Executor() {
    for(auto const& matrix: pool) {
        delete matrix;
    }
}

void Executor::setup_from_file(std::ifstream &file_stream) {
    /*
    Инициализировать исполнителя матрицами из файла.
    Контракт обмена жесткий
    Первая строчка, два целых числа через пробел - размер и количество матриц
    Далее без раздления следуют матрицы
    */
    uint32_t size, quantity;
    file_stream >> size >> quantity;
    file_stream.ignore();
    std::cout << "Size matrix: " << size << '\n';
    std::cout << "Quantity matrix: " << quantity << '\n';

    // Резервируем память для матриц
    this->pool.reserve(quantity);
    this->results.reserve(quantity);

    // В цикле создаём объекты матрицы и загружаем данные из файла
    for (uint32_t i = 0; i < quantity; i++) {
        Matrix *matrix = new Matrix(size);
        file_stream >> *matrix;
        this->pool.push_back(matrix);
        // std::cout << *this->pool[i] << '\n';
    }

    // Указавыаем сколько памяти занимают все матрицы
    memory = quantity * this->pool[0]->size();

    return;
}

void Executor::execute() {
    /*
    Выполняем нашу операцию над всеми матрицами
    */
    // std::cout << "Run execute" << '\n';
    // std::cout << "Size pool " << this->pool.size() << '\n';

    auto start = high_resolution_clock::now();

    // OpenMP
    uint32_t i;
    #if defined(_OPENMP)
        uint32_t nthreads = 16;
        #pragma omp parallel for private(i) num_threads(nthreads)
    #endif
    for (i = 0; i < this->pool.size(); i++) {
        this->results.push_back(this->pool[i]->count_under_diagonal());
        // std::cout << this->results[i] << '\n';
    }

    auto stop = high_resolution_clock::now();
    this->duration = duration_cast<milliseconds>(stop - start);

    return;
}

void Executor::save_result_to_file(std::ofstream& file_stream) {
    // std::cout << "Run to save" << '\n';
    for (auto const& result: this->results) {
        // std::cout << result << '\n';
        file_stream << result << '\n';
    }

    std::ostringstream string_stream;
    string_stream << "Took: " << this->duration.count() << " ms" << '\n';
    string_stream << "Memory: " << this->memory << " bytes" << '\n';
    std::cout << string_stream.str();
    file_stream << string_stream.str();

    return;
}
