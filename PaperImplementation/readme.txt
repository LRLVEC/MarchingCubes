This is the demo program running on GPU using CUDA. 
It extracts and renders the analytical Cayley surfaces, which is defined in a 3D field 
by evaluating the algebraic surface, f(x, y, z) = 1 - 16xyz -4x^2  - 4y^2 - 4z^2, at an input 3D voxel grid.



Requirements:  
1). An Nvidia Kepler GPU (at least GeForce GTX 650), or a better one, with at least Compute Capability 3.0;
     see this link, FYI: https://developer.nvidia.com/cuda-gpus
2). The latest GPU driver (version 347.62), or later ones.
3). 64-bit Windows7 operation system. 
 


Usage: You can directly launch the provided batch files (*.bat) by double-clicking them; or you can use the command line as below.


Command line Usage:   "MC gridSizeLog2.x gridSizeLog2.y gridSizeLog2.z"   or: 
"MC  gridSizeLog2"
, where gridSizeLog2 indicates the voxel resolution in power of two (along XYZ)


Example usage:  "MC 9 9 9"     or     "MC 9"
, which indicates the voxel resolution is  9th power of two, i.e., 512,  (along XYZ).

If you have a GeForce GTX Titan GPU (with 6 GB video memory or more), you can try "MC2048x2048x4096.bat"  by double-clicking it.
If your GPU does not have enough memory to run a larger dataset, try to use a smaller input resolution.



Keyboard operation: 
"+"    increase isoValue
"-"    decrease isoValue
" "    press spacebar (blank key) to enable animation.

Or you can right click to get menu operations.




