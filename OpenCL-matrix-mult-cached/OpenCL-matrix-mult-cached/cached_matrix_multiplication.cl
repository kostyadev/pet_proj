#if !defined(SUB_SIZE) && !defined(WPT)
/**
 * This kernel function efficiently multiplies two matrices a[M,K] and b[K,N] by "naive" way
 */
__kernel void multiplyMatrices(__global int* a,
	__global int* b,
	__global int* c,
	const int M,
	const int N,
	const int K)
{
	//Get work - item identifiers.
	int colIndex = get_global_id(0);
	int rowIndex = get_global_id(1);

	//Compute element c[rowIndex, colIndex].
	int sum = 0;
	for (int k = 0; k < K; k++) {
		sum += a[rowIndex*K + k] * b[k*N + colIndex];
	}
	// save result
	c[(rowIndex * N) + colIndex] = sum;
}

#endif

#if defined(SUB_SIZE) && !defined(WPT)
/**
 * This kernel function efficiently multiplies two matrices a[M,K] and b[K,N] by multiply submatrix
 */
__kernel void multiplyMatrices(__global int* a,
	__global int* b,
	__global int* c,
	const int M,
	const int N,
	const int K)
{
	// Get work-item identifiers.
	int colIndex = get_local_id(0);
	int rowIndex = get_local_id(1);
	int globalColIndex = get_global_id(0);
	int globalRowIndex = get_global_id(1);
	int index = (globalRowIndex * N) + globalColIndex;

	// Create submatrices that will cache the matrices A and B in local memory.
	__local int aSub[SUB_SIZE][SUB_SIZE];
	__local int bSub[SUB_SIZE][SUB_SIZE];

	// Initialize accumulator register.
	int sum = 0;

	// Loop over all submatrices.
	const int nSub = K / SUB_SIZE;
	for (int s = 0; s < nSub; s++)
	{
		// Load submatrices into local memory.
		const int sCol = SUB_SIZE * s + colIndex;
		const int sRow = SUB_SIZE * s + rowIndex;
		aSub[rowIndex][colIndex] = a[globalRowIndex * K + sCol];
		bSub[rowIndex][colIndex] = b[sRow * N + globalColIndex];

		// Synchronize all work-items in this work-group.
		barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single submatrix.
		for (int k = 0; k < SUB_SIZE; k++) {
			sum += aSub[rowIndex][k] * bSub[k][colIndex];
		}

		// Synchronize all work-items in this work-group.
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Store the final result in the matrix C.
	c[index] = sum;
}

#endif

#if defined(SUB_SIZE) && defined(WPT)
/**
 * This kernel function efficiently multiplies two matrices a[M,K] and b[K,N]
 * by caching submatrices from those input matrices in the device local memory
 * and use "work per thread" method.
 */
__kernel void multiplyMatrices(__global int* a,
                                    __global int* b,
                                    __global int* c,
                                    const int M,
                                    const int N, 
                                    const int K)
{
	const int RTS = SUB_SIZE / WPT;
	// Get work-item identifiers.
	const int row = get_local_id(0); // Local row ID (max: SUB_SIZE)
	const int col = get_local_id(1); // Local col ID (max: SUB_SIZE/WPT == RTS)
	const int globalRow = SUB_SIZE * get_group_id(0) + row; // Row ID of C (0..M)
	const int globalCol = SUB_SIZE * get_group_id(1) + col; // Col ID of C (0..N)

	// Create submatrices that will cache the matrices A and B in local memory.
    __local int aSub[SUB_SIZE][SUB_SIZE];
    __local int bSub[SUB_SIZE][SUB_SIZE];

	// Initialize accumulator 
	float acc[WPT];
	for (int w = 0; w < WPT; w++) 
	{
		acc[w] = 0.0f;
	}

	// Loop over all submatrices.
    const int nSub = K / SUB_SIZE;
    for(int s = 0; s < nSub; s++)
	{
		// Load one tile of A and B into local memory
		for (int w = 0; w < WPT; w++) 
		{
			const int tiledRow = SUB_SIZE * s + row;
			const int tiledCol = SUB_SIZE * s + col;
			aSub[col + w * RTS][row] = a[(tiledCol + w * RTS)*M + globalRow];
			bSub[col + w * RTS][row] = b[(globalCol + w * RTS)*K + tiledRow];
		}

		// Synchronize all work-items in this work-group.
        barrier(CLK_LOCAL_MEM_FENCE);

		// Perform the computation for a single tile
		for (int k = 0; k < SUB_SIZE; k++)
		{
			for (int w = 0; w < WPT; w++) 
			{
				acc[w] += aSub[k][row] * bSub[col + w * RTS][k];
			}
		}

		// Synchronize all work-items in this work-group.
        barrier(CLK_LOCAL_MEM_FENCE);
    }

	// Store the final results in C
	for (int w = 0; w < WPT; w++) 
	{
		c[(globalCol + w * RTS)*M + globalRow] = acc[w];
	}
}
#endif