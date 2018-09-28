// This kernal takes a float array as input, and outputs a reduced float array, with the 0th element
// in the array being the sum of all values in the input

__kernel void ReduceFloatArray(__global const float* A, __global float* B) {
	// Get the current worker thread
	int id = get_global_id(0);

	// Get the number of workers
	int N = get_global_size(0);

	// Copy each element in the input array to the output array
	B[id] = A[id];

	// Wait for the copy to complete before moving on
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Loop through each worker and increase by the stride each time (2 in this case)
	for (int i = 1; i < N; i *= 2) 
	{
		if (!(id % (i * 2)) && ((id + i) < N))
		{ 
			B[id] += B[id + i];
		}

		// Wait for the addition to complete before continuing
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

// This kernel checks the value of 2 floats, and if the first is larger than the second, swaps them using
// a temporary variable.
void Swap(__global float* A, __global float* B) 
{
	// Check if float A is larger than float B then swap
	if (*A > *B) 
	{
		int t = *A; *A = *B; *B = t;
	}
}

// In order to find the minimum and maximum values for a float array, they must first be sorted using a Bitonic Sort algorthm.
// 
__kernel void MinMaxSort(__global float* A, __global float* B) 
{
	// Get the current worker thread
	int id = get_global_id(0);

	// Get the number of workers
	int N = get_global_size(0);

	for (int i = 0; i < N; i += 2) 
	{
		// Check if the current thread is even and i is less than the size of the workers
		if (id % 2 == 1 && id + 1 < N)
		{
			// Swap the current value with the next element
			Swap(&A[id], &A[id + 1]);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		// Check if the current thread is odd and i less than the size of the workers
		if (id % 2 == 0 && id + 1 < N)
		{
			// Swap the current values with the next element
			Swap(&A[id], &A[id + 1]);
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		// Copy each element of the input float array to the output array
		B[id] = A[id];
	}
}

__kernel void Histogram(__global const int* A, __global int* H) {
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}