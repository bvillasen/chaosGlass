#include <stdint.h>
#include <cuda.h>

// #define HEIGHT 256
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void mappingLogistic_kernel( const int nWidth, const int nHeight, cudaP xMin, cudaP xMax, cudaP yMin, cudaP yMax,cudaP *startingPoints, cudaP *graphPoints ){
  int tid = blockIdx.x + threadIdx.x*gridDim.x;
  
  __shared__ unsigned int mappedPoints[ %(HEIGHT)s ];
  mappedPoints[threadIdx.x] = 0;
  __syncthreads();
  
  cudaP val = startingPoints[ threadIdx.x + blockIdx.x*blockDim.x];
  cudaP k = (xMax - xMin)/(nWidth-1)*blockIdx.x + xMin;
  int nValues = 1000;
  cudaP yFactor = double(nHeight)/(yMax-yMin);
  int yPix;
  for (int i=0; i<100000; i++) val =  k*val*(1-val); //Tranciente
  for (int i=0; i<nValues; i++ ){
    if ( val>=yMin and val <=yMax){
      yPix = int((val-yMin)*yFactor);
      if (yPix<nHeight and yPix>=0) mappedPoints[yPix] += 1;
    }
    val =  k*val*(1-val);
  }
  cudaP value;
  if (mappedPoints[threadIdx.x]>=1) value = log(double(mappedPoints[threadIdx.x]));
  else value = 0.0f;
  graphPoints[tid] = value;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void mask_kernel( int xMin, int xMax, int yMin, int yMax, int *maskPoints){
  int t_x = blockIdx.x*blockDim.x + threadIdx.x;
  int t_y = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_x + t_y*blockDim.x*gridDim.x;
  
  
  int val;
  if ( (t_x<xMax && t_x>xMin) && (t_y<yMax && t_y>yMin) )  val = 0;
  else val = 1;
  maskPoints[tid] = val;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void plot_kernel( int jMin, int jMax, int iMin, int iMax, cudaP *graphPoints, int *maskPoints, cudaP *plotData){
  int t_x = blockIdx.x*blockDim.x + threadIdx.x;
  int t_y = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_x + t_y*blockDim.x*gridDim.x;
  cudaP val=graphPoints[tid];
  if ( (t_x>=jMin and t_x<jMax) and (t_y>=iMin and t_y<iMax) ) val = 1-val; 
  plotData[tid] = val;
}