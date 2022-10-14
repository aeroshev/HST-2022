#ifndef READMATRIX_H
#define READMATRIX_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int **readmatrix(size_t *rows, size_t *cols, const char *filename);

#endif