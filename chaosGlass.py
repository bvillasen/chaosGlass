# 2D visualisation tool for chaotic mappings
# made by Bruno Villasenor
# contact me at: bvillasen@gmail.com
# personal web page:  https://bvillasen.webs.com
# github: https://github.com/bvillasen

#To run you need these complementary files: CUDAchaosGlass.cu, animation2D.py, cudaTools.py
#you can find them in my github: 
#                               https://github.com/bvillasen/animation2D
#                               https://github.com/bvillasen/tools

import sys, time, os
import numpy as np
import pylab as plt
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom

#Add Modules from other directories
currentDirectory = os.getcwd()
parentDirectory = currentDirectory[:currentDirectory.rfind("/")]
toolsDirectory = parentDirectory + "/tools"
animation2DDirectory = parentDirectory + "/animation2D"
sys.path.extend( [toolsDirectory, animation2DDirectory] )

import animation2D
from cudaTools import setCudaDevice, getFreeMemory, gpuArray2DtocudaArray

precision = "double"

useDevice = None
for option in sys.argv:
  if option == "gray": animation2D.usingGrayScale = True
  if option == "float": precision = "float"
  if option.find("device=") != -1: useDevice = int(option[-1]) 

npPrcsn = np.float64 if precision=="double" else np.float32

nWidth = 1024
nHeight = 256*2
nData = nWidth*nHeight

#Set upper and lower limits for plotting
xMin, xMax = 2.8, 4.0
yMin, yMax = 0., 1.
jMin, jMax = 10000, -1
iMin, iMax = 10000, -1

#Initialize openGL
animation2D.nWidth = nWidth
animation2D.nHeight = nHeight
animation2D.initGL()

#set thread grid for CUDA kernels
block_size_x, block_size_y  = 16, 16   #hardcoded, tune to your needs
gridx = nWidth // block_size_x + 1 * ( nWidth % block_size_x != 0 )
gridy = nHeight // block_size_y + 1 * ( nHeight % block_size_y != 0 )
grid2D = (gridx, gridy, 1)
block2D = (block_size_x, block_size_y, 1)
mapBlock = ( nHeight, 1, 1 )
mapGrid = ( nWidth, 1, 1 )

#initialize pyCUDA context 
cudaDevice = setCudaDevice( devN=useDevice, usingAnimation=True )

#Read and compile CUDA code
print "Compiling CUDA code"
cudaCodeFile = open("CUDAchaosGlass.cu","r")
cudaCodeStringRaw = cudaCodeFile.read() 
cudaCodeString = (cudaCodeStringRaw %{"HEIGHT":mapBlock[0]}).replace("cudaP", precision)
cudaCode = SourceModule(cudaCodeString)
mappingLogisticKernel = cudaCode.get_function('mappingLogistic_kernel')
maskKernel = cudaCode.get_function('mask_kernel')
plotKernel = cudaCode.get_function('plot_kernel')
########################################################################
from pycuda.elementwise import ElementwiseKernel
########################################################################
linearDouble = ElementwiseKernel(arguments="double a, double b, double *input, double *output",
				operation = "output[i] = a*input[i] + b ")

#Initialize all gpu data
print "Initializing Data"
initialMemory = getFreeMemory( show=True )  
random_d = curandom.rand((nData), dtype=npPrcsn) 
graphPoints_d= gpuarray.to_gpu( np.zeros([nData], dtype=npPrcsn) ) 	
#For plotting
maskPoints_h = np.ones(nData).astype(np.int32)
maskPoints_d = gpuarray.to_gpu( maskPoints_h )
plotData_d = gpuarray.to_gpu( np.zeros([nData], dtype=npPrcsn) )
finalMemory = getFreeMemory( show=False )
print " Total Global Memory Used: {0} Mbytes".format(float(initialMemory-finalMemory)/1e6) 

def replot():
  global xMin, xMax, yMin, yMax
  global jMin, jMax, iMin, iMax
  global random_d
  jMin, jMax = animation2D.jMin, animation2D.jMax
  iMin, iMax = animation2D.iMin, animation2D.iMax
  xMin += (xMax-xMin)*(float(jMin)/nWidth)
  xMax -= (xMax-xMin)*(float(nWidth-jMax)/nWidth)
  yMin += (yMax-yMin)*(float(iMin)/nHeight)
  yMax -= (yMax-yMin)*(float(nHeight-iMax)/nHeight)
  print "Reploting: ( {0} , {1} , {2} , {3} )".format(xMin, xMax, yMin, yMax)
  start, end = cuda.Event(), cuda.Event()
  start.record()
  random_d = curandom.rand((nData), dtype=npPrcsn)
  mappingLogisticKernel( np.int32(nWidth), np.int32(nHeight), npPrcsn(xMin), npPrcsn(xMax), npPrcsn(yMin), npPrcsn(yMax), random_d, graphPoints_d, grid=mapGrid, block=mapBlock )
  normalize( graphPoints_d )
  end.record()
  end.synchronize()
  print " Map Calculated in: %f secs\n" %( start.time_till(end)*1e-3)
  animation2D.jMin, animation2D.jMax = 10000, -1
  animation2D.iMin, animation2D.iMax = 10000, -1
  maskFunc()
  
def caosGlassFrame():
  #mappingKernel( np.int32(nWidth), np.int32(nHeight), npPrcsn(xMin), npPrcsn(xMax), npPrcsn(yMin), npPrcsn(yMax), random_d, graphPoints_d, grid=mapGrid, block=mapBlock )
  return 0

def normalize( data ):
  maxVal = gpuarray.max(data).get()
  linearDouble(1./maxVal, np.float64(0.), data, data )
  

def maskFunc():
  jMin, jMax = np.int32(animation2D.jMin), np.int32(animation2D.jMax)
  iMin, iMax = np.int32(animation2D.iMin), np.int32(animation2D.iMax)
  plotKernel( jMin, jMax, iMin, iMax, graphPoints_d, maskPoints_d, plotData_d, grid=grid2D, block=block2D  )

mappingLogisticKernel( np.int32(nWidth), np.int32(nHeight), npPrcsn(xMin), npPrcsn(xMax), npPrcsn(yMin), npPrcsn(yMax), random_d, graphPoints_d, grid=mapGrid, block=mapBlock )
normalize( graphPoints_d )
plotKernel( np.int32(jMin), np.int32(jMax), np.int32(iMin), np.int32(iMax), graphPoints_d, maskPoints_d, plotData_d, grid=grid2D, block=block2D  )

#configure animation2D stepFunction and plotData
animation2D.stepFunc = caosGlassFrame
if precision=="double": animation2D.usingDouble = True
animation2D.plotData_d = plotData_d
animation2D.backgroundType = "square"
animation2D.mouseMaskFunc = maskFunc
animation2D.replotFunc = replot

#run animation
animation2D.animate()
