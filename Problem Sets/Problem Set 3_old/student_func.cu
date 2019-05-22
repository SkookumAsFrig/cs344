/* Udacity Homework 3
   HDR Tone-mapping

   Background HDR
   ==============

   A High Dynamic Range (HDR) image contains a wider variation of intensity
   and color than is allowed by the RGB format with 1 byte per channel that we
   have used in the previous assignment.

   To store this extra information we use single precision floating point for
   each channel.  This allows for an extremely wide range of intensity values.

   In the image for this assignment, the inside of church with light coming in
   through stained glass windows, the raw input floating point values for the
   channels range from 0 to 275.  But the mean is .41 and 98% of the values are
   less than 3!  This means that certain areas (the windows) are extremely bright
   compared to everywhere else.  If we linearly map this [0-275] range into the
   [0-255] range that we have been using then most values will be mapped to zero!
   The only thing we will be able to see are the very brightest areas - the
   windows - everything else will appear pitch black.

   The problem is that although we have cameras capable of recording the wide
   range of intensity that exists in the real world our monitors are not capable
   of displaying them.  Our eyes are also quite capable of observing a much wider
   range of intensities than our image formats / monitors are capable of
   displaying.

   Tone-mapping is a process that transforms the intensities in the image so that
   the brightest values aren't nearly so far away from the mean.  That way when
   we transform the values into [0-255] we can actually see the entire image.
   There are many ways to perform this process and it is as much an art as a
   science - there is no single "right" answer.  In this homework we will
   implement one possible technique.

   Background Chrominance-Luminance
   ================================

   The RGB space that we have been using to represent images can be thought of as
   one possible set of axes spanning a three dimensional space of color.  We
   sometimes choose other axes to represent this space because they make certain
   operations more convenient.

   Another possible way of representing a color image is to separate the color
   information (chromaticity) from the brightness information.  There are
   multiple different methods for doing this - a common one during the analog
   television days was known as Chrominance-Luminance or YUV.

   We choose to represent the image in this way so that we can remap only the
   intensity channel and then recombine the new intensity values with the color
   information to form the final image.

   Old TV signals used to be transmitted in this way so that black & white
   televisions could display the luminance channel while color televisions would
   display all three of the channels.


   Tone-mapping
   ============

   In this assignment we are going to transform the luminance channel (actually
   the log of the luminance, but this is unimportant for the parts of the
   algorithm that you will be implementing) by compressing its range to [0, 1].
   To do this we need the cumulative distribution of the luminance values.

   Example
   -------

input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
min / max / range: 0 / 9 / 9

histo with 3 bins: [4 7 3]

cdf : [4 11 14]


Your task is to calculate this cumulative distribution by following these
steps.

*/

#include "utils.h"
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

__global__ void shmem_reduce_min (const float* const d_in,
		const size_t numRows, const size_t numCols,
		float* d_result_min)
{
	extern __shared__ float temp_dlogLum[];
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);//This is the thread's id within global threads

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;//This is the thread's id within local threads
	if (!(myID.x >= numCols || myID.y >= numRows)){
		const int myID1D = myID.y * numCols + myID.x;//Fill the block's dlogLum local memory with block image data
		temp_dlogLum[tid] = d_in[myID1D];
	}else{
		temp_dlogLum[tid] = FLT_MAX;
	}
	__syncthreads();
	//pow(2, ceilf(log2f(blockDim.x*blockDim.y) - 1))
	for (unsigned int i = 2<<__float2int_rn(ceilf(log2f(blockDim.x*blockDim.y) - 2)); i>0; i>>=1){
		if (tid < i){
			temp_dlogLum[tid] = min(temp_dlogLum[tid], temp_dlogLum[tid + i]);
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_result_min[blockIdx.x + blockIdx.y * gridDim.x] = temp_dlogLum[0];
	}
}

__global__ void shmem_reduce_max (const float* const d_in,
		const size_t numRows, const size_t numCols,
		float* d_result_min)//, uint8_t sw)
{
	extern __shared__ float temp_dlogLum[];
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);

	const int tid = threadIdx.x + threadIdx.y * blockDim.x;
	if (!(myID.x >= numCols || myID.y >= numRows)){
		const int myID1D = myID.y * numCols + myID.x;
		temp_dlogLum[tid] = d_in[myID1D];
		/*if(sw){
		  printf("shemem value is %f, at tid %d and myID %d\n", temp_dlogLum[tid], tid, myID1D);
		  }*/
	}else{
		temp_dlogLum[tid] = FLT_MIN;
	}
	__syncthreads();
	//printf("%d\n", 2<<__float2int_rn(ceilf(log2f(blockDim.x*blockDim.y) - 2)));
	for (unsigned int i = 2<<__float2int_rn(ceilf(log2f(blockDim.x*blockDim.y) - 2)); i>0; i>>=1){
		if (tid < i){
			/*
			   if(tid == 44 || tid == 20 || tid == 8 || tid == 2 || tid == 1 || tid == 0){
			   printf("tid is at problem loc %d, values are small = %f, big = %f\n",
			   tid, temp_dlogLum[tid], temp_dlogLum[tid + i]);
			   }
			 */
			temp_dlogLum[tid] = max(temp_dlogLum[tid], temp_dlogLum[tid + i]);
		}
		__syncthreads();
	}

	if (tid == 0)
	{
		d_result_min[blockIdx.x + blockIdx.y * gridDim.x] = temp_dlogLum[0];
	}
}

__global__ void shmem_histo (const float* const d_in,
		const size_t numItems, const float lumMin,
		const float lumRange, const size_t numRows,
		const size_t numCols, const size_t numBins,
		int *d_globalbins)
{
	const int2 myID = make_int2(threadIdx.x + blockDim.x * blockIdx.x,
			threadIdx.y + blockDim.y * blockIdx.y);
	const int myID1D = myID.y*gridDim.x*blockDim.x + myID.x;

	//const int tid = threadIdx.x + threadIdx.y * blockDim.x;
	//const int thnum = blockDim.x*blockDim.y;

	for (int i = 0; i<numItems; i++){
		int indim = myID1D*numItems + i;
		if (indim<numRows*numCols){
			int bin = (d_in[indim] - lumMin) / lumRange * numBins;
			atomicAdd(&(d_globalbins[bin]), 1);
		}
	}

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



const int thread_x = 32;
const int thread_y = 32;

void your_histogram_and_prefixsum(const float* const d_logLuminance,
		unsigned int* const d_cdf,
		float &min_logLum,
		float &max_logLum,
		const size_t numRows,
		const size_t numCols,
		const size_t numBins)
{
	//TODO
	/*Here are the steps you need to implement
	  1) find the minimum and maximum value in the input logLuminance channel
	  store in min_logLum and max_logLum
	  2) subtract them to find the range
	  3) generate a histogram of all the values in the logLuminance channel using
	  the formula: bin = (lum[i] - lumMin) / lumRange * numBins
	  4) Perform an exclusive scan (prefix sum) on the histogram to get
	  the cumulative distribution of luminance values (this should go in the
	  incoming d_cdf pointer which already has been allocated for you)       */
	int numpix = numRows*numCols;
	float *d_inter, *d_min, *d_max;
	checkCudaErrors(cudaMalloc(&d_inter, numpix*sizeof(float)));
	checkCudaErrors(cudaMemset(d_inter, 126, numpix*sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_min, sizeof(float)));
	checkCudaErrors(cudaMemset(d_min, 0, sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_max, sizeof(float)));
	checkCudaErrors(cudaMemset(d_max, 0, sizeof(float)));

	const int gridx = ceil(numCols/thread_x);
	const int gridy = ceil(numRows/thread_y);
	const dim3 gridSizeMM(gridx, gridy, 1);
	const dim3 blockSizeMM(thread_x, thread_y, 1);

	//printf("gridx is %d, gridy is %d\n", gridx, gridy);

	const size_t shmemSZ = (thread_x*thread_y) * sizeof(float);

	shmem_reduce_min<<<gridSizeMM, blockSizeMM, shmemSZ>>>
		(d_logLuminance, numRows, numCols, d_inter);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	/*
	   float h_inter[numpix];
	   checkCudaErrors(cudaMemcpy(h_inter, d_inter, numpix*sizeof(float), cudaMemcpyDeviceToHost));
	   for (int j=0; j<256; j++){
	   printf("minval is %f at %d\n", h_inter[j], j);
	   }
	 */
	const int grids = 1;
	const dim3 blocks = gridSizeMM;

	shmem_reduce_min<<<grids, blocks, shmemSZ>>>
		(d_inter, gridy, gridx, d_min);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&min_logLum, d_min, sizeof(float), cudaMemcpyDeviceToHost));
	printf("smallest number is: %f\n", min_logLum);
	//////////////////////////////////////////////////////////////////////MIN DONE, NOW MAX

	checkCudaErrors(cudaMemset(d_inter, 240, numpix*sizeof(float)));
	shmem_reduce_max<<<gridSizeMM, blockSizeMM, shmemSZ>>>
		(d_logLuminance, numRows, numCols, d_inter);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	/*
	   checkCudaErrors(cudaMemcpy(h_inter, d_inter, numpix*sizeof(float), cudaMemcpyDeviceToHost));
	   for (int j=0; j<400; j++){
	   printf("maxval is %f at %d\n", h_inter[j], j);
	   }
	 */
	shmem_reduce_max<<<grids, blocks, shmemSZ>>>
		(d_inter, gridy, gridx, d_max);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(&max_logLum, d_max, sizeof(float), cudaMemcpyDeviceToHost));
	printf("biggest number is: %f\n", max_logLum);
	///////////////////////////////////////////////////////////////////////MAX DONE
	const float dlogRange = max_logLum - min_logLum;

	/*
	   shmem_histo (const float* const d_in,
	   const size_t numItems, const float lumMin,
	   const float lumRange, const size_t numRows,
	   const size_t numCols, const size_t numBins,
	   int* d_globalbins)//, uint8_t sw)
	 */
	int *d_bins;
	checkCudaErrors(cudaMalloc(&d_bins, numBins*sizeof(int)));
	checkCudaErrors(cudaMemset(d_bins, 0, numBins*sizeof(int)));

	const int hthreadx = 32;
	const int hthready = 32;
	const int hgridx = 1;
	const int hgridy = 1;
	const dim3 hgridSize(hgridx, hgridy, 1);
	const dim3 hblockSize(hthreadx, hthready, 1);
	const int numPerTh = ceil(numRows*numCols/(hthreadx*hthready));

	shmem_histo<<<hgridSize, hblockSize>>>
		(d_logLuminance, numPerTh, min_logLum, dlogRange, numRows, numCols, numBins, d_bins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	/*
	int h_bins[numBins];
	checkCudaErrors(cudaMemcpy(&h_bins[0], d_bins, numBins*sizeof(int), cudaMemcpyDeviceToHost));
	
	float rsum = 0.f;
	for (int j=0; j<numBins; j++){
		printf("%d bin count is %d\n", j, h_bins[j]);
		rsum += h_bins[j];
	}
	printf("sum is %f\n", rsum);
	*/

	checkCudaErrors(cudaMemcpy(d_cdf, d_bins, numBins*sizeof(int), cudaMemcpyDeviceToDevice));
	
	blelloch_scan<<<grids, blockSizeMM>>>(d_cdf, numBins);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
