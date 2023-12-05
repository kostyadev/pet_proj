#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <thread>
#include <sstream>
#include <chrono>

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

cl::Device getDefaultDevice();        // Return the first device found in this OpenCL platform.
void initializeDevice();              // Inicialize device and compile kernel code.
void seqMultiplyMatrices(int* a,
	int* b,
	int* c,
	const int M,
	const int N,
	const int K); // Sequentially performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatricesNaive(int* a,
	int* b,
	int* c,
	const int M,
	const int N,
	const int K); // Parallelly performs the operation c[M,N] = a[M,K] * b[K,N] (naive).
void parMultiplyMatricesSubmatrix(int* a,
	int* b,
	int* c,
	const int M,
	const int N,
	const int K); // Parallelly performs the operation c[M,N] = a[M,K] * b[K,N] (submatrix).
void parMultiplyMatricesSubmatrixWpt(int* a,
	int* b,
	int* c,
	const int M,
	const int N,
	const int K); // Parallelly performs the operation c[M,N] = a[M,K] * b[K,N] (submatrix + WRT).
void parMultiplyMatrices_CPU(int* a,
	int* b,
	int* c,
	const int M,
	const int N,
	const int K); // Parallelly performs the operation (by CPU) c[M,N] = a[M,K] * b[K,N].
bool checkEquality(int* c1,
	int* c2,
	const int M,
	const int N);      // Check if the matrices c1 and c2 are equal.

// =================================================================
// ------------------------ Global Variables ------------------------
// =================================================================

cl::Context context;                // The context which holds the device.    
cl::Device device;                  // The device where the kernel will run.

cl::Program programNaive;                // The programs that will run on the device (naive).    
cl::Program programSubMatrix;			// The programs that will run on the device (submatrix).    
cl::Program programSubMatrixWpt;		// The programs that will run on the device (submatrix + WPT).    

const size_t WPT = 8;				// The work per thread 
const size_t SUB_SIZE = 16;		   // The default value of work group.

// =================================================================
// ------------------------- Main Function -------------------------
// =================================================================

int main() 
{
	// Create auxiliary variables.
	const int EXECUTIONS = 10;

	// Prepare input constants related to the dimensions of the matrices.
	const int M = 1 << 10;
	const int N = 1 << 10;
	const int K = 1 << 10;

	// Prepare input matrices A and B.
	const size_t ROWS_A = M;
	const size_t COLS_A = K;
	std::vector<int> a(ROWS_A * COLS_A, 3);

	const size_t ROWS_B = K;
	const size_t COLS_B = N;
	std::vector<int> b(ROWS_B * COLS_B, 5);

	// Prepare sequential and parallel output matrices.
	const size_t ROWS_C = M;
	const size_t COLS_C = N;
	std::vector<int> cs(ROWS_C * COLS_C);
	std::vector<int> cp(ROWS_C * COLS_C);
	std::vector<int> cps(ROWS_C * COLS_C);
	std::vector<int> cpsw(ROWS_C * COLS_C);
	std::vector<int> cp_cpu(ROWS_C * COLS_C);

	// Sequentially multiply matrices.
	double seqCpuMs = 0.0f;
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < EXECUTIONS; i++) {
			seqMultiplyMatrices(a.data(), b.data(), cs.data(), M, N, K);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		seqCpuMs = std::chrono::duration<double, std::milli>(t_end - t_start).count() / EXECUTIONS;
	}

	// Parallel multiply matrices on CPU.
	double parCpuMs = 0.0f;
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < EXECUTIONS; i++) {
			parMultiplyMatrices_CPU(a.data(), b.data(), cp_cpu.data(), M, N, K);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		parCpuMs = std::chrono::duration<double, std::milli>(t_end - t_start).count() / EXECUTIONS;
	}

	// Initialize OpenCL device.
	initializeDevice();

	// Parallelly multiply matrices
	double parGpuNaiveMs = 0.0f;
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < EXECUTIONS; i++) {
			parMultiplyMatricesNaive(a.data(), b.data(), cp.data(), M, N, K);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		parGpuNaiveMs = std::chrono::duration<double, std::milli>(t_end - t_start).count() / EXECUTIONS;
	}

	double parGpuSubMatrMs = 0.0f;
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < EXECUTIONS; i++) {
			parMultiplyMatricesSubmatrix(a.data(), b.data(), cps.data(), M, N, K);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		parGpuSubMatrMs = std::chrono::duration<double, std::milli>(t_end - t_start).count() / EXECUTIONS;
	}

	double parGpuSubMatrWptMs = 0.0f;
	{
		auto t_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < EXECUTIONS; i++) {
			parMultiplyMatricesSubmatrixWpt(a.data(), b.data(), cpsw.data(), M, N, K);
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		parGpuSubMatrWptMs = std::chrono::duration<double, std::milli>(t_end - t_start).count() / EXECUTIONS;
	}

	// Check if outputs are equal.
	bool equal1 = checkEquality(cs.data(), cp.data(), ROWS_C, COLS_C);
	bool equal2 = checkEquality(cs.data(), cps.data(), ROWS_C, COLS_C);
	bool equal3 = checkEquality(cs.data(), cpsw.data(), ROWS_C, COLS_C);
	bool equal4 = checkEquality(cs.data(), cp_cpu.data(), ROWS_C, COLS_C);

	// Print results.
	std::cout << "Status: " << (equal1 && equal2 && equal3 && equal4 ? "SUCCESS!" : "FAILED!") << std::endl;
	std::cout << "Results: \n\tA[0] = " << a[0] << "\n\tB[0] = " << b[0] << "\n\tC[0] = " << cp[0] << std::endl;
	std::cout << "Mean execution time: \n\tSequential CPU: " << seqCpuMs << " ms;\n\tParallel (naive): " << parGpuNaiveMs 
		<< " ms;\n\tParallelSubmatrix: " << parGpuSubMatrMs
		<< " ms;\n\tParallelSubmatrixWpt: " << parGpuSubMatrWptMs
		<< " ms;\n\tParallel_CPU: " << parCpuMs << " ms." << std::endl;
	return 0;
}

// =================================================================
// ---------------------- Secondary Functions ----------------------
// =================================================================

// Return the first device found in this OpenCL platform.
cl::Device getDefaultDevice() 
{
	 // Search for all the OpenCL platforms available and check
	 // if there are any.
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty()) {
		std::cerr << "No platforms found!" << std::endl;
		exit(1);
	}


	 // Search for all the devices on the first platform and check if
	 // there are any available.
	auto platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

	if (devices.empty()) 
	{
		std::cerr << "No devices found!" << std::endl;
		exit(1);
	}

	// Return the first device found.
	return devices.front();
}

// Inicialize device and compile kernel code.
void initializeDevice() 
{
	// Select the first available device.
	device = getDefaultDevice();

	// Read OpenCL kernel file as a string.
	std::ifstream kernel_file("cached_matrix_multiplication.cl");
	std::string src(std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

	// Compile kernel program which will run on the device.
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));
	context = cl::Context(device);

	{
		programNaive = cl::Program(context, sources);
		auto err = programNaive.build();
		if (err != CL_BUILD_SUCCESS) {
			std::cerr << "Error!\nBuild Status: " << programNaive.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
				<< "\nBuild Log:\t " << programNaive.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
			exit(1);
		}
	}

	{
		programSubMatrix = cl::Program(context, sources);
		std::stringstream ss;
		ss <<" -D SUB_SIZE=" << SUB_SIZE; // defines for kernel function
		auto err = programSubMatrix.build(ss.str().c_str());
		if (err != CL_BUILD_SUCCESS) {
			std::cerr << "Error!\nBuild Status: " << programSubMatrix.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
				<< "\nBuild Log:\t " << programSubMatrix.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
			exit(1);
		}
	}

	{
		programSubMatrixWpt = cl::Program(context, sources);
		std::stringstream ss;
		ss << "-D WPT=" << WPT << " -D SUB_SIZE=" << SUB_SIZE; // defines for kernel function
		auto err = programSubMatrixWpt.build(ss.str().c_str());
		if (err != CL_BUILD_SUCCESS) {
			std::cerr << "Error!\nBuild Status: " << programSubMatrixWpt.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device)
				<< "\nBuild Log:\t " << programSubMatrixWpt.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
			exit(1);
		}
	}
}

// Sequentially performs the operation c[M,N] = a[M,K] * b[K,N].
void seqMultiplyMatrices(int* a, int* b, int* c,
	const int M,
	const int N,
	const int K) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			int sum = 0;
			for (int k = 0; k < K; k++) {
				sum += a[i*K + k] * b[j + k * N];
			}
			c[i*N + j] = sum;
		}
	}
}

// Parallelly performs the operation (on CPU) c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatrices_CPU(int* a, int* b, int* c,
	const int M,
	const int N,
	const int K) 
{
	const auto thrCnt = std::thread::hardware_concurrency();
	std::vector<std::thread> workers;
	const int rod = N % thrCnt;
	const unsigned int bColCntPerThread = (N - rod) / thrCnt;
	for (int thrId = 0; thrId < thrCnt; ++thrId)
	{
		workers.push_back(std::thread([thrId, thrCnt, bColCntPerThread, a, b, c, M, N, K]()
			{
				std::vector<int> bColBuf(K);
				auto bColbegIdx = thrId * bColCntPerThread;
				bool isLastThread = (thrId == thrCnt - 1);
				auto bColEndIdx = isLastThread ? N : bColbegIdx + bColCntPerThread;
				//each thread iterate some amount of columns in B matrix 
				for (auto bColIdx = bColbegIdx; bColIdx < bColEndIdx; ++bColIdx)
				{
					//fill column buffer for B matr
					for (int i = 0; i < K; ++i)
					{
						bColBuf[i] = b[i * N + bColIdx];
					}
					//multiply all rows of A to current column of B
					for (int aRowIdx = 0; aRowIdx < M; ++aRowIdx)
					{
						int tmp = 0;
						for (int aColIdx = 0; aColIdx < K; ++aColIdx)
						{
							tmp += a[aRowIdx * M + aColIdx] * bColBuf[aColIdx];
						}
						c[aRowIdx * M + bColIdx] = tmp;
					}
				}
			}));
	}

	for (std::thread& t : workers)
	{
		t.join();
	}
	workers.clear();
}

// Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatricesNaive(int* a, int* b, int* c,
	const int M,
	const int N,
	const int K) 
{
	// Create buffers and allocate memory on the device.
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, M * K * sizeof(int), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, K * N * sizeof(int), b);
	cl::Buffer cBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));

	// Set kernel arguments.
	cl::Kernel kernel(programNaive, "multiplyMatrices");
	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, &M);
	kernel.setArg(4, &N);
	kernel.setArg(5, &K);

	// Execute the kernel function and collect its result.
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, M));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
	queue.finish();
}

// Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatricesSubmatrix(int* a, int* b, int* c,
	const int M,
	const int N,
	const int K) 
{
	// Create buffers and allocate memory on the device.
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, M * K * sizeof(int), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, K * N * sizeof(int), b);
	cl::Buffer cBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));

	// Set kernel arguments.
	cl::Kernel kernel(programSubMatrix, "multiplyMatrices");
	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, &M);
	kernel.setArg(4, &N);
	kernel.setArg(5, &K);

	// Execute the kernel function and collect its result.
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, M), cl::NDRange(SUB_SIZE, SUB_SIZE));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
	queue.finish();
}

// Parallelly performs the operation c[M,N] = a[M,K] * b[K,N].
void parMultiplyMatricesSubmatrixWpt(int* a, int* b, int* c,
	const int M,
	const int N,
	const int K) 
{
	// Create buffers and allocate memory on the device.
	cl::Buffer aBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, M * K * sizeof(int), a);
	cl::Buffer bBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, K * N * sizeof(int), b);
	cl::Buffer cBuf(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, M * N * sizeof(int));

	// Set kernel arguments.
	cl::Kernel kernel(programSubMatrixWpt, "multiplyMatrices");
	kernel.setArg(0, aBuf);
	kernel.setArg(1, bBuf);
	kernel.setArg(2, cBuf);
	kernel.setArg(3, &M);
	kernel.setArg(4, &N);
	kernel.setArg(5, &K);

	// Execute the kernel function and collect its result.
	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N, M / WPT), cl::NDRange(SUB_SIZE, SUB_SIZE / WPT));
	queue.enqueueReadBuffer(cBuf, CL_TRUE, 0, M * N * sizeof(int), c);
	queue.finish();
}

// Check if the matrices C1 and C2 are equal.
bool checkEquality(int* c1, int* c2,
	const int M,
	const int N)
{
	for (int i = 0; i < M*N; i++) {
		if (c1[i] != c2[i]) {
			return false;
		}
	}
	return true;
}