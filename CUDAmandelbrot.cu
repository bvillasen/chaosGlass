#include <stdint.h>
#include <cuda.h>

#define PI 3.14159265359

// #define HEIGHT 256
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <pycuda-complex.hpp>
typedef   pycuda::complex<double> pyComplex;

__device__ float norma(pyComplex z){
  return norm(z);
}


__global__ void mandelbrot_kernel(double xMin, double xMax, double yMin, double yMax, int L, double *M) {
  int n_x = blockDim.x*gridDim.x;
  //int n_y = blockDim.y*gridDim.y;
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  int idy = threadIdx.y + blockDim.y*blockIdx.y;
  //int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = idy*n_x+idx;

  double x0 = (xMin + xMax)/2;
  double y0 = (yMin + yMax)/2;
  double side = xMax - xMin;
  
  float delta = side/n_x;
  pyComplex c( x0-side/2.+delta*idx,y0-side/2.+delta*idy);
  pyComplex z( x0-side/2.+delta*idx,y0-side/2.+delta*idy);

  double h = 0;
  int L1= 1700;
  float R = 2.0;

  while( h<L1 && norma(z)<R){
    z=z*z+c;
    h+=1;
  }
  M[threadId]=log(h + 1);

}
/*
__global__ void mandelbrot_kernel( const int nWidth, const int nHeight, cudaP xMin, cudaP xMax,
					cudaP yMin, cudaP yMax,cudaP *startingPoints, cudaP *graphPoints ){
  int tid = blockIdx.x + threadIdx.x*gridDim.x;
  
  __shared__ unsigned int mappedPoints[ %(HEIGHT)s ];
  mappedPoints[threadIdx.x] = 0;
  __syncthreads();
  
  cudaP val = startingPoints[ threadIdx.x + blockIdx.x*blockDim.x];
  cudaP k = (xMax - xMin)/(nWidth-1)*blockIdx.x + xMin;
  int nValues = 1500;
  cudaP yFactor = cudaP(nHeight)/(yMax-yMin);
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
  if (mappedPoints[threadIdx.x]>=1) value = log(cudaP(mappedPoints[threadIdx.x]));
  else value = 0.0f;
  graphPoints[tid] = value;
}*/


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