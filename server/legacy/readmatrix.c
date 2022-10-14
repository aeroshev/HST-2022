#include "readmatrix.h"


int **readmatrix(size_t *rows, size_t *cols, const char *filename)
{
    if(rows == NULL || cols == NULL || filename == NULL)
        return NULL;

    *rows = 0;
    *cols = 0;

    FILE *fp = fopen(filename, "r");

    if(fp == NULL)
    {
        fprintf(stderr, "could not open %s: %s\n", filename, strerror(errno));
        return NULL;
    }

    int **matrix = NULL, **tmp;

    char line[1024];

    while(fgets(line, sizeof line, fp))
    {
        if(*cols == 0)
        {
            // determine the size of the columns based on
            // the first row
            char *scan = line;
            int dummy;
            int offset = 0;
            while(sscanf(scan, "%d%n", &dummy, &offset) == 1)
            {
                scan += offset;
                (*cols)++;
            }
        }

        tmp = realloc(matrix, (*rows + 1) * sizeof *matrix);

        if(tmp == NULL)
        {
            fclose(fp);
            return matrix; // return all you've parsed so far
        }

        matrix = tmp;

        matrix[*rows] = calloc(*cols, sizeof *matrix[*rows]);

        if(matrix[*rows] == NULL)
        {
            fclose(fp);
            if(*rows == 0) // failed in the first row, free everything
            {
                fclose(fp);
                free(matrix);
                return NULL;
            }

            return matrix; // return all you've parsed so far
        }

        int offset = 0;
        char *scan = line;
        for(size_t j = 0; j < *cols; ++j)
        {
            if(sscanf(scan, "%d%n", matrix[*rows] + j, &offset) == 1)
                scan += offset;
            else
                matrix[*rows][j] = 0; // could not read, set cell to 0
        }

        // incrementing rows
        (*rows)++;
    }

    fclose(fp);

    return matrix;
}
