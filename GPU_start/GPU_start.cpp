#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <cmath>
#include <ctime>
#include <locale.h>
#include <iostream>
#include <omp.h>
#include <ctype.h>
#include <algorithm>
#include <vector>
#include <assert.h>

#include "GPU.h"

using namespace std;

double* GPU_solve(double *A, int *B, int *C, double *right, double *diag, int non_zero, int size)
{
	gpu_solver obj;
	double *solution = new double[size];
	solution = obj.GPU_gradient_solver(A, B, C, right, diag, non_zero, size);
	return solution;
}

void main()
{
	clock_t init = clock();
	int size = 16000;
	vector<double> A_v;
	vector<int> B_v;
	vector<int> C_v;
	vector<double> right_v;
	vector<double> diag_v;

	srand(time(NULL));
	double ** matrix;
	matrix = new double*[size];

	//Matrix generation block will be deleted in the future
	//Matrix generation block beginning
	//Memory allocation for matrix
	for (int i = 0; i < size; i++)
	{
		matrix[i] = new double[size];
	}
	//Filling by zeroes
	for (int i = 0; i < size; i++)
	{

		for (int j = 0; j < size; j++)
		{
			matrix[i][j] = 0;
		}
	}
	//Filling collateral diagonals
	for (int i = 0; i < size - 1; i++)
	{
		matrix[i][i + 1] = rand() % 10 + 1;
		matrix[i + 1][i] = matrix[i][i + 1];
	}
	//Just view of matrix
	/*for (int i = 0; i < size; i++)
	{
	cout << endl;
	for (int j = 0; j < size; j++)
	cout << matrix[i][j] << "  ";
	}*/
	cout << endl << endl;
	//Filling central diagonal
	for (int i = 0; i < size; i++)
	{
		diag_v.push_back(rand() % 10 + 1);
	}

	double sum = 0;
	//Transfer to SCR
	for (int i = 0; i < size; i++)
	{
		C_v.push_back(A_v.size());
		for (int j = 0; j < size; j++)
		{

			if (matrix[i][j] != 0)
			{
				if (i == size - 1)
				{
					sum++;
				}
				A_v.push_back(matrix[i][j]);
				B_v.push_back(j);
			}
		}

	}
	C_v.push_back(A_v.size());
	//Filling vector b
	for (int i = 0; i < size; i++)
	{
		right_v.push_back(rand() % 10 + 1);
	}
	//Matrix generation block ending

	for (int i = 0; i < size; i++)
	{
		delete matrix[i];
	}
	delete[] matrix;

	//Memory allocation for pointer that will be sent to solver
	double *A = new double[A_v.size()];
	int *B = new int[B_v.size()];
	int *C = new int[C_v.size()];
	double * right = new double[right_v.size()];
	double* diag = new double[diag_v.size()];
	double *sol = new double[size];

	//Container2Pointer migration
	for (int i = 0; i < diag_v.size(); i++)
	{
		diag[i] = diag_v[i];
	}
	for (int i = 0; i < A_v.size(); i++)
	{
		A[i] = A_v[i];
	}
	for (int i = 0; i < B_v.size(); i++)
	{
		B[i] = B_v[i];
	}
	for (int i = 0; i < C_v.size(); i++)
	{
		C[i] = C_v[i];
	}
	for (int i = 0; i < right_v.size(); i++)
	{
		right[i] = right_v[i];
	}
	int nnz = A_v.size();
	A_v.clear();
	B_v.clear();
	C_v.clear();
	right_v.clear();
	diag_v.clear();

	clock_t benit = clock();
	cout << "INIT TIME:  " << double(benit - init) / 1000.0 << endl;
	clock_t int1 = clock();
	//Solving
	sol = GPU_solve(A, B, C, right, diag,nnz, size);
	clock_t int2 = clock();
	cout << "TIME:  " << double(int2 - int1) / 1000.0 << endl;
	/*for (int i = 0; i < size; i++)
	{
		cout << sol[i] << " ";
	}*/

	system("pause");
}