---
published: true
---
# Getting ready for Cuda

Using Aras [Toy Path tracer](https://github.com/aras-p/ToyPathTracer), I decided to use [this commit](https://github.com/aras-p/ToyPathTracer/commits/c2376c1e45acc38af5d13b38819da0d6e91f8933) as a starting point as it contains the following improvements:
- explicit random state instead of local-thread storage
- multiple fixes to the renderer to get it closer to Mitsuba

In addition I made a lot of modifications to simplify the code and make it easier for me to use Cuda:
- I removed Unity, C#, Mac, and multi-threading related code
- I simplified the Windows main to be a simple console application that runs a renders a predefined number of frames into a .png image (thanks to [stb_image_write](https://github.com/nothings/stb/blob/master/stb_image_write.h)).

Here is [a link](https://github.com/voxel-tracer/CudaPathTracer/tree/01-code-prep) to the full source code
