// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

const size_t block_x = 32;
const size_t block_y = 32;

	__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
		unsigned char* const greyImage,
		int numRows, int numCols)
{
	//TODO
	//Fill in the kernel to convert from color to greyscale
	//the mapping from components of a uchar4 to RGBA is:
	// .x -> R ; .y -> G ; .z -> B ; .w -> A
	//
	//The output (greyImage) at each pixel should be the result of
	//applying the formula: output = .299f * R + .587f * G + .114f * B;
	//Note: We will be ignoring the alpha channel for this conversion

	//First create a mapping from the 2D block and grid locations
	//to an absolute 2D location in the image, then use that to
	//calculate a 1D offset
	size_t blkid_x = blockIdx.x;
	size_t blkid_y = blockIdx.y;
	size_t thid_x = threadIdx.x;
	size_t thid_y = threadIdx.y;
	int tpbx = numCols/block_x + 1;
	//int tpby = numRows/block_y;
	size_t realid = (blkid_x + blkid_y*tpbx)*block_x*block_y + thid_x + thid_y*block_x;
	if (realid < numRows*numCols){
		uchar4 rgba = rgbaImage[realid];
		float channelSum = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
		greyImage[realid] = channelSum;
	}
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
		unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
	//You must fill in the correct sizes for the blockSize and gridSize
	//currently only one block with one thread is being launched
	const dim3 blockSize(block_x, block_y, 1);  //TODO
	int gridx = numCols/block_x + 1;
	int gridy = numRows/block_y + 1;
	printf("grid x is %d\ngrid y is %d\n", gridx, gridy);
	const dim3 gridSize(gridx, gridy, 1);  //TODO

	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
