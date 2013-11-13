#include <stdint.h>
#include <cuda.h>

#define HEIGHT 256
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
__global__ void mapping_kernel( const int nWidth, const int nHeight, double xMin, double xMax, double yMin, double yMax,double *startingPoints, double *graphPoints ){
  int t_x = blockIdx.x*blockDim.x + threadIdx.x;
  int t_y = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_x + t_y*blockDim.x*gridDim.x;
  
  __shared__ double mappedPoints[HEIGHT];
  mappedPoints[t_y] = 0;
  __syncthreads();
  
  double val = startingPoints[tid];
  double k = (xMax - xMin)/(nWidth-1)*t_x + xMin;
  int nIterations = 100000;
  int nValues = 100;
  double yFactor = nHeight/(yMax-yMin);
  int yPix;
  for (int i=0; i<nIterations; i++) val =  k*val*(1-val); //Tranciente
  for (int i=0; i<nValues; i++ ){
    if ( val>=yMin and val <=yMax){
      yPix = int((val-yMin)*yFactor);
      if (yPix<nHeight and yPix>=0) mappedPoints[yPix] += 1;
    }
    val =  k*val*(1-val);
  }
  
  double value;
  if (mappedPoints[t_y]>0) value = 0.99f;
  else value = 0.2f;
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
__global__ void plot_kernel( int jMin, int jMax, int iMin, int iMax, double *graphPoints, int *maskPoints, double *plotData){
  int t_x = blockIdx.x*blockDim.x + threadIdx.x;
  int t_y = blockIdx.y*blockDim.y + threadIdx.y;
  int tid = t_x + t_y*blockDim.x*gridDim.x;
  double val;
  if ( (t_x>=jMin and t_x<jMax) and (t_y>=iMin and t_y<iMax) ) val = 0.5; 
  else val = 1;
  plotData[tid] = val*graphPoints[tid];
}