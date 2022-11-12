#include <vector>
#include <fstream>
#include <tuple>
#include <mpi.h>

using namespace std;

tuple<int, int> read_matrix(ifstream&, vector<float>&);
float computeGPU(vector<float>&, int, int);
void my_abort(int);
