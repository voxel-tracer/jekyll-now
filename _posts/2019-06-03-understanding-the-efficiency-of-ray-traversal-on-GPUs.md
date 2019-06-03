---
published: true
---
## How to build "Understanding the Efficiency of Ray Traversal on GPUs"

As I was looking for ways to improve my [CUDA path tracer](link to my dla renderer) I found [this](https://research.nvidia.com/publication/understanding-efficiency-ray-traversal-gpus) excellent paper by Aila, Laine, and Karras. The original source code is still available in [Google Code Archive](https://code.google.com/archive/p/understanding-the-efficiency-of-ray-traversal-on-gpus/).

While it was not that complicated to load it in VS2017 and get it to build with CUDA v9.2. Trying to run the program fails with the following error:
```
No appropriate pixel format found!
```
Luckily someone already figured out how to fix this. The following Github [repository](https://github.com/AlanIWBFT/gpu-ray-traversal) has an updated version CUDA v8.0, and it also fixes the pixel format issue in Windows 10.

The repository doesn't contain the scenes' .obj file, most likely because they take too much space. You just need to copy the whole scenes folder from the original source code.

Trying to load the VS solution as is doesn't work though, as it is expecting CUDA v8.0 to be installed. The fix is simple, just edit framework.vcxproj and rt.vcxproj and replace all occurrences of "CUDA 8.0" with "CUDA 9.2" (or whatever version you have). 

You can then build the project, just make sure you pick "Release" and "x64" as the build's configuration and platform. Once the build finishes, you can run the program either in interactive mode (default) or in batch mode

Now you can run the program as is in interactive mode or batch mode. Here is an example of running the program in batch mode with the San Miguel scene:
```
./rt_x64_Release.exe benchmark --mesh=scenes/rt/sanmiguel/sanmiguel.obj --camera=Yciwz1oRQmz/Xvsm005CwjHx/b70nx18tVI7005frY108Y/:x/v3/z100
```

You can get the camera string for all existing scenes from the source code of rt/App.cpp
