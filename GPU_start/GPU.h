#pragma once

using namespace std;

#ifdef MATHFUNCSDLL_EXPORTS
#define MATHFUNCSDLL_API __declspec(dllexport) 
#else
#define MATHFUNCSDLL_API __declspec(dllimport) 
#endif

class gpu_solver {

public:

	MATHFUNCSDLL_API double* GPU_gradient_solver(double *A, int *B, int *C, double *right, double *diag, int non_zero, int size); //Main Function

};