//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

Note: ascending order == smallest to largest

Each score is associated with a position, when you sort the scores, you must
also move the positions accordingly.

Implementing Parallel Radix Sort with CUDA
==========================================

The basic idea is to construct a histogram on each pass of how many of each
"digit" there are.   Then we scan this histogram so that we know where to put
the output of each digit.  For example, the first 1 must come after all the
0s so we have to know how many 0s there are to be able to start moving 1s
into the correct position.

1) Histogram of the number of occurrences of each digit
2) Exclusive Prefix Sum of Histogram
3) Determine relative offset of each digit
For example [0 0 1 1 0 0 1]
->  [0 1 0 1 2 3 2]
4) Combine the results of steps 2 & 3 to determine the final
output location for each element and move it there

LSB Radix sort is an out-of-place sort and you will need to ping-pong values
between the input and output buffers we have provided.  Make sure the final
sorted results end up in the output buffer!  Hint: You may need to do a copy
at the end.

 */

const unsigned int bitdepth = 8*sizeof(unsigned int);
const unsigned int numBits = 8;
const unsigned int numBins = 1 << numBits;

__global__ void histo_offset (unsigned int* d_inputVals,
		unsigned int* d_binHistogram,
		unsigned int* d_offsetvec,
		const size_t numElems, unsigned int i)
{
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	const int myID1D = myID.y*gridDim.x*blockDim.x + myID.x;

	if (myID1D >= numElems)
		return;

	unsigned int currVal = d_inputVals[myID1D];

	unsigned int mask = (numBins - 1) << i;
	unsigned int bin = (currVal & mask) >> i;
	d_offsetvec[myID1D] = atomicAdd(& d_binHistogram[bin], 1);
}


__global__ void blelloch_scan (unsigned int *d_bins_io,
		const size_t numBins)
{
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	const int myID1D = myID.y*gridDim.x*blockDim.x + myID.x;

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;
	const int thnum = blockDim.x*blockDim.y;

	for (unsigned int i = 2; i<=numBins; i<<=1){
		if ((tid+1)%i == 0){
			unsigned int step = i>>1;
			d_bins_io[myID1D] += d_bins_io[myID1D-step];
		}
		__syncthreads();
	}

	if (tid == thnum-1) d_bins_io[myID1D] = 0;
	__syncthreads();

	for (unsigned int j = numBins; j>0; j>>=1){
		if ((tid+1)%j == 0){
			unsigned int step2 = j>>1;
			unsigned int right = d_bins_io[myID1D];
			d_bins_io[myID1D] += d_bins_io[myID1D-step2];
			d_bins_io[myID1D-step2] = right;
		}
		__syncthreads();
	}

}

__global__ void swapLocs (unsigned int* d_inputVals, unsigned int* d_inputPos,
		unsigned int* d_outputVals, unsigned int* d_outputPos,
		unsigned int* d_binScan, unsigned int* d_offsetvec,
		const size_t numElems, unsigned int i)
{
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
                        threadIdx.y + blockDim.y * blockIdx.y);
        const int myID1D = myID.y*gridDim.x*blockDim.x + myID.x;

	if (myID1D >= numElems)
                return;

        unsigned int currVal = d_inputVals[myID1D];
        unsigned int currPos = d_inputPos[myID1D];

	unsigned int mask = (numBins - 1) << i;
	unsigned int bin = (currVal & mask) >> i;

	unsigned int newidx = d_binScan[bin] + d_offsetvec[myID1D];
	/*if (newidx>=numElems)
                printf("newidx overflowed: %d\n", newidx);*/
	d_outputVals[newidx] = currVal;
	d_outputPos[newidx] = currPos;

}

__global__ void old_prototype (unsigned int* d_inputVals, unsigned int* const d_inputPos,
		unsigned int* d_outputVals, unsigned int* d_outputPos,
		unsigned int* d_binHistogram,
		unsigned int* d_offsetvec,
		unsigned int* d_binScan,
		const size_t numElems)
{
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	const int myID1D = myID.y*gridDim.x*blockDim.x + myID.x;

	if (myID1D >= numElems)
		return;

	unsigned int currVal = d_inputVals[myID1D];
	unsigned int currPos = d_inputPos[myID1D];

	for (unsigned int i=0;i<bitdepth;i+=numBits){
		unsigned int mask = (numBins - 1) << i;
		unsigned int bin = (currVal & mask) >> i;
		d_offsetvec[myID1D] = atomicAdd(& d_binHistogram[bin], 1);
		__syncthreads();
		if (myID1D == 0){
			d_binScan[0] = 0;
			for (unsigned int j=1;j<numBins;j++){
				d_binScan[j] = d_binScan[j-1] + d_binHistogram[j-1];
				printf("d_bins at %d is %d\n", j, d_binScan[j]);
				d_binHistogram[j-1] = 0;
			}
			d_binHistogram[numBins-1] = 0;
		}
		__syncthreads();
		unsigned int newidx = d_binScan[bin] + d_offsetvec[myID1D];
		//if (newidx>=numElems)
		//printf("newidx overflowed: %d\n", newidx);
		//d_outputVals[newidx] = currVal;
		//d_outputPos[newidx] = currPos;

		__syncthreads();
		currVal = d_outputVals[myID1D];
		currPos = d_outputPos[myID1D];
	}
}


void your_sort(unsigned int* const d_inputVals,
		unsigned int* const d_inputPos,
		unsigned int* const d_outputVals,
		unsigned int* const d_outputPos,
		const size_t numElems)
{
	//TODO
	//PUT YOUR SORT HERE
	unsigned int *d_binHisto, *d_offset, *d_binScan, *d_swapBufV, *d_swapBufP;
	printf("numBins is %d\n", numBins);
	printf("numElems is %d\n", (int) numElems);
	checkCudaErrors(cudaMalloc(&d_binHisto, numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_offset, numElems*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_binScan, numBins*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_swapBufV, numElems*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_swapBufP, numElems*sizeof(unsigned int)));

	const int thread_x = 32;
	const int thread_y = 32;
	const int grids = ceil(sqrt(numElems/(thread_x*thread_y)));
	const dim3 blockSize(thread_x, thread_y, 1);
	const dim3 gridSize(grids, grids, 1);

	checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
	
	for (unsigned int i=0;i<bitdepth;i+=numBits)
	{
		checkCudaErrors(cudaMemset(d_binHisto, 0, numBins*sizeof(unsigned int)));
		histo_offset<<<gridSize, blockSize>>>(d_outputVals, d_binHisto, d_offset, numElems, i);
		
		checkCudaErrors(cudaMemcpy(d_binScan, d_binHisto, numBins*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		blelloch_scan<<<1, numBins>>>(d_binScan, numBins);
		
		cudaDeviceSynchronize();
		
		swapLocs<<<gridSize, blockSize>>>(d_outputVals, d_outputPos, d_swapBufV, d_swapBufP, d_binScan, d_offset, numElems, i);

		checkCudaErrors(cudaMemcpy(d_outputVals, d_swapBufV, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputPos, d_swapBufP, numElems*sizeof(unsigned int), cudaMemcpyDeviceToDevice));

		cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
		/*
		   unsigned int *h_binScan = new unsigned int[numBins];
		   checkCudaErrors(cudaMemcpy(h_binScan, d_binHisto, numBins*sizeof(unsigned int), cudaMemcpyDeviceToHost));
		   unsigned int sumBins = 0;
		   for (int j=0;j<numBins;j++){
		   sumBins+=h_binScan[j];
		   }
		   printf("sumBins = %d", sumBins);
		 */
	}
}
