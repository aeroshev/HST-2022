#include "matrix_counter.h"


int count_under_diagonal(int **matrix, unsigned int size) {
    int sum = 0;
    unsigned int left_side;

    for (unsigned int row = 0; row < size; row++) {
        left_side = row + 1;
        for (unsigned int column = left_side; column < size; column++) {
            sum += matrix[row][column];
        }
    }

    return sum;
}
