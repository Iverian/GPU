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

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>

#include <cuda.h>
#include <cusparse.h>
#include <cublas.h>
#include <cublas_api.h>

#include <helper_cuda.h>
#include <helper_cuda_drvapi.h>
#include <helper_functions.h>
#include <helper_cusolver.h>

#include "GPU.h"

vector<double> GPU_mult(vector<double> vec, int size, int *nnz, vector<double> diag, int gpu_amount, double **d_A, int **d_B, int ** d_C);
int* split(int gpu_amount, vector<double> A, vector<int> B, vector<int> C, int size, int non_zero, double **d_A, int ** d_B, int **d_C);
vector<double> gradient_method(vector<double> A, vector<int> B, vector<int> C, vector<double>right, int size, int gpu, vector<double> diag);

double* gpu_solver::GPU_gradient_solver(double *A, int *B, int *C, double *right, double *diag, int non_zero, int size) //Main Function
{
	vector<double> A_v;
	vector<int> B_v;
	vector<int> C_v;
	vector<double> right_v;
	vector<double> diag_v;
	int gpu;
	checkCudaErrors(cudaGetDeviceCount(&gpu));

	for (int i = 0; i < non_zero; i++)
	{
		A_v.push_back(A[i]);
	}

	for (int i = 0; i < non_zero; i++)
	{
		B_v.push_back(B[i]);
	}

	for (int i = 0; i < size + 1; i++)
	{
		C_v.push_back(C[i]);
	}

	for (int i = 0; i < size; i++)
	{
		diag_v.push_back(diag[i]);
	}

	for (int i = 0; i < size; i++)
	{
		right_v.push_back(right[i]);
	}

	/*clock_t start = clock();
	cout << endl << "Rezult:" << endl;*/
	vector<double> solve = gradient_method(A_v, B_v, C_v, right_v, size - 1, gpu, diag_v);
	/*clock_t finish = clock();
	cout << "TIME:  " << double(finish - start) / 1000.0 << endl;*/
	//double * Result_final = &solve[0];
	return &solve[0];
}

double dot_product(vector <double> A, vector<double> B)
{
	double rezult = 0;
	if (A.size() != B.size())
	{
		cout << "Wrong size of vector" << endl;
		return 0;
	}
	else
	{
		for (int i = 0; i < A.size(); i++)
		{
			rezult += (A[i] * B[i]);
		}
		return rezult;
	}
}

vector <double> vector_on_number(vector<double> A, double value)
{
	for (int i = 0; i < A.size(); i++)
		A[i] = A[i] * value;
	return A;
}

vector<double> sum_vector(vector<double> A, vector <double> B)
{
	if (A.size() != B.size())
	{
		cout << "Wrong size of vector" << endl;
		return A;
	}
	else
	{
		for (int i = 0; i < A.size(); i++)
		{
			A[i] = A[i] + B[i];
		}
		return A;
	}
}

vector<double> raznost_vector(vector<double> A, vector <double> B)
{
	if (A.size() != B.size())
	{
		cout << "Wrong size of vector" << endl;
		return A;
	}
	else
	{
		for (int i = 0; i < A.size(); i++)
		{
			A[i] = A[i] - B[i];
		}
		return A;
	}
}

vector<double> gradient_method(vector<double> A, vector<int> B, vector<int> C, vector<double> right, int size, int gpu, vector<double> diag)
{
	double ** d_A = new  double *[gpu];
	int ** d_B = new int *[gpu];
	int ** d_C = new int *[gpu];

	int *temp = new int[gpu];

	vector<double> r0;
	vector<double> x0(size + 1, 0);
	vector<double> x_k;
	vector<double> z0;
	vector<double> z_k;
	vector<double> r_k;
	double epsilon = pow(10, -10);
	r0 = right;// x0 ={0...}
	z0 = r0;
	r_k = r0;
	double a_k;
	double b_k;
	double ny;
	double checking = 0;
	bool fg = true;
	int step = 0;
	double gpu_time=0;
	clock_t int1 = clock();
	vector<double> ch;
	temp = split(gpu, A, B, C, size, A.size(), d_A, d_B, d_C);
	clock_t int2 = clock();
	cout << "SPLIT TIME:  " << double(int2 - int1) / 1000.0 << endl;
	do
	{
		if (fg == false)
		{
			r0 = r_k;
			x0 = x_k;
			z0 = z_k;
		}
		clock_t gpu_time1 = clock();
		ch = GPU_mult(z0, size + 1, temp, diag, gpu, d_A, d_B, d_C);
		clock_t gpu_time2 = clock();
		gpu_time += double(gpu_time2 - gpu_time1);
		a_k = dot_product(r0, r0) / dot_product(ch, z0);
		x_k = sum_vector(x0, vector_on_number(z0, a_k));
		r_k = raznost_vector(r0, vector_on_number(ch, a_k));
		b_k = dot_product(r_k, r_k) / dot_product(r0, r0);
		z_k = sum_vector(r_k, vector_on_number(z0, b_k));
		fg = false;
		step++;
		checking = sqrt(dot_product(r_k, r_k)) / sqrt(dot_product(right, right));
		//ch.clear();
	} while (((sqrt(dot_product(r_k, r_k)) / sqrt(dot_product(right, right))) >= epsilon));
	cout << "GPU TIME:  " << gpu_time / 1000.0 << endl;
	
	cout << checking << endl;

	for (int number = 0; number < gpu; number++)
	{
		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaFree(d_A[number]));
		checkCudaErrors(cudaFree(d_B[number]));
		checkCudaErrors(cudaFree(d_C[number]));
	}

	//clock_t int2 = clock();
	cout << endl << "Step" << endl << step << endl;
	//cout << endl << "Solution:" << endl;
	/*for (int i = 0; i < size + 1; i++)
	{
	cout << x_k[i] << " ";
	}
	cout << endl;*/
	delete[] temp;

	return x_k;
}

vector<double> sopr_mult(vector<double> on, vector <double> A, vector <int> B, vector <int> C)
{
	vector <double> rez(on.size(), 0);
	double sum = 0;
	int m = -1;
	{

		for (int i = 0; i < C.size() - 1; i++)
		{
			m++;
			for (int j = C[i]; j < C[i + 1]; j++)
			{
				sum += A[j] * on[B[j]];
			}
			rez[m] = sum;

			sum = 0;
		}
	}
	return rez;
}

int return_string(int number, vector<int> C)
{
	int i = 0;
	while (C[i] <= number)
		i++;
	return i;
}

int* split(int gpu_amount, vector<double> A, vector<int> B, vector<int> C, int size, int non_zero, double **d_A, int ** d_B, int **d_C) // Костляво
{
	int mod = non_zero / gpu_amount; // уходит на все 
	int rest = non_zero - mod*(gpu_amount - 1); //уходит на последнюю 
	int first_position;
	int last_position;
	int first_string;
	int last_string;
	vector<int>::iterator bit1;
	vector<int>::iterator bit2;
	vector<double>::iterator it1;
	vector<double>::iterator it2;

	int *temp = new int[gpu_amount];

	/*cout << endl;
	for (int i = 0; i < A.size(); i++)
	{
	cout << A[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < B.size(); i++)
	{
	cout << B[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < C.size(); i++)
	{
	cout << C[i] << " ";
	}*/

	vector<double> A_;
	vector<int> B_;
	vector<int> C_;

	for (int number = 0; number < gpu_amount; number++)
	{
		if (number == gpu_amount - 1)
		{
			first_position = number*mod;// n 
			last_position = non_zero - 1;//k 
			first_string = return_string(number*mod, C) - 1; //i 
			last_string = return_string(non_zero - 1, C) - 1;//j 

			A_.assign(rest + first_string + size - last_string, 0); // definition 
			it1 = A.begin();
			it2 = A_.begin();

			copy(it1 + first_position, it1 + last_position + 1, it2 + first_string); //00...00 A 00...000 

			B_.assign(first_string + rest + size - last_string, 0);

			bit1 = B.begin();
			bit2 = B_.begin();

			copy(bit1 + first_position, bit1 + last_position + 1, bit2 + first_string); //00...00 B 000 

			for (int i = 0; i < first_string; i++) //0123..B..000 
				B_[i] = i;
			C_.assign(size + 2, 0);

			for (int i = 0; i < first_string; i++) //0123..C..000 
				C_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				C_[count] = C[count] - first_position + first_string;
				if (C[count] - first_position < 0) C_[count] = first_string;
			}
			C_[size + 1] = A_.size();
		}
		else
		{
			first_position = number*mod;// n 
			last_position = (number + 1)*mod - 1;//k 
			first_string = return_string(number*mod, C) - 1; //i 
			last_string = return_string((number + 1)*mod - 1, C) - 1;//j 

			A_.assign(mod + first_string + size - last_string, 0); // definition 
			it1 = A.begin();
			it2 = A_.begin();

			copy(it1 + first_position, it1 + last_position + 1, it2 + first_string); //00...00 A 00...000 

			B_.assign(first_string + mod + size - last_string, 0);

			bit1 = B.begin();
			bit2 = B_.begin();

			copy(bit1 + first_position, bit1 + last_position + 1, bit2 + first_string); //00...00 B 000 

			for (int i = 0; i < first_string; i++) //0123..B..000 
				B_[i] = i;
			int inn = 1;
			for (int i = first_string + mod; i < size - last_string + first_string + mod; i++)
			{
				B_[i] = last_string + inn;
				inn++;
			}
			C_.assign(size + 2, 0);

			for (int i = 0; i < first_string; i++) //0123..C..000 
				C_[i] = i;
			for (int count = first_string; count <= last_string; count++)
			{
				C_[count] = C[count] - first_position + first_string;
				if (C[count] - first_position < 0) C_[count] = first_string;
			}
			int l = 1;
			for (int i = last_string + 1; i < size + 1; i++) //0123..C..n.. 
			{
				C_[i] = first_string + last_position - first_position + l;
				l++;
			}
			C_[size + 1] = A_.size();
		}

		temp[number] = A_.size();

		checkCudaErrors(cudaSetDevice(number));
		checkCudaErrors(cudaMalloc((void **)&d_A[number], sizeof(double)*A_.size()));
		checkCudaErrors(cudaMalloc((void **)&d_B[number], sizeof(int)*B_.size()));
		checkCudaErrors(cudaMalloc((void **)&d_C[number], sizeof(int)*C_.size()));
		checkCudaErrors(cudaMemcpy(d_A[number], &A_[0], sizeof(double)*A_.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_B[number], &B_[0], sizeof(int)*B_.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_C[number], &C_[0], sizeof(int)*C_.size(), cudaMemcpyHostToDevice));

		/*cout << endl;
		for (int i = 0; i < A_.size(); i++)
		{
		cout << A_[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < B_.size(); i++)
		{
		cout << B_[i] << " ";
		}
		cout << endl;
		for (int i = 0; i < C_.size(); i++)
		{
		cout << C_[i] << " ";
		}*/

		A_.clear();
		B_.clear();
		C_.clear();
	}
	return temp;
}

vector<double> GPU_mult(vector<double> vec, int size, int *nnz, vector<double> diag, int gpu_amount, double **d_A, int **d_B, int **d_C)
{
	vector<double> rezult(size, 0);
	double **pipe = new double*[gpu_amount];
	for (int i = 0; i < gpu_amount; i++)
	{
		pipe[i] = new double[size];
	}
	//size == vec.size()
	double ** rez_p = new  double *[gpu_amount];
	cusparseHandle_t handle = NULL;
	cusparseMatDescr_t Adescr = NULL;
	double *one = new double;
	*one = 1.0;
	double *zero = new double;
	*zero = 0.0;
	double *x_d;
	omp_set_num_threads(gpu_amount);
#pragma omp parallel for private (pointer,rez_p,handle,Adescr,vec_pt)
	{
		for (int number = 0; number < gpu_amount; number++)
		{
			checkCudaErrors(cudaSetDevice(number));
			checkCudaErrors(cudaMalloc((void **)&x_d, sizeof(double)*size));
			checkCudaErrors(cudaMemcpy(x_d, &vec[0], sizeof(double)*size, cudaMemcpyHostToDevice));
			checkCudaErrors(cusparseCreate(&handle));
			checkCudaErrors(cusparseCreateMatDescr(&Adescr));
			checkCudaErrors(cusparseSetMatType(Adescr, CUSPARSE_MATRIX_TYPE_GENERAL));
			checkCudaErrors(cusparseSetMatIndexBase(Adescr, CUSPARSE_INDEX_BASE_ZERO));

			checkCudaErrors(cudaMalloc((void **)&rez_p[number], sizeof(double)*size));

			checkCudaErrors(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				size, size, nnz[number], one,
				Adescr,
				d_A[number],
				d_C[number], d_B[number],
				x_d, zero,
				rez_p[number]));

			checkCudaErrors(cudaMemcpy(pipe[number], rez_p[number], sizeof(double)*size, cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(rez_p[number]));
			checkCudaErrors(cudaFree(x_d));
		}
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < gpu_amount; j++)
		{
			rezult[i] += pipe[j][i];
		}
	}

	for (int i = 0; i < size; i++)
	{
		rezult[i] += diag[i] * vec[i];
	}
	/*cout << endl;
	for (int i = 0; i < size; i++)
	{
	cout << donat[i] << " ";
	}*/
	for (int i = 0; i < gpu_amount; i++)
	{
		delete pipe[i];
	}
	delete[] pipe;
	delete zero;
	delete one;
	delete[] rez_p;
	return rezult;
}
