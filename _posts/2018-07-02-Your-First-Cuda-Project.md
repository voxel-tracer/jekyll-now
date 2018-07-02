---
published: true
---
## How to install Visual Studio 2017, Cuda, and get your first project running

### Install Visual Studio 2017 Community edition
You can download the installer from [here](https://visualstudio.microsoft.com/downloads/). As of this writing, latest available version is 15.7.4 (we'll need this information later).

Installation should be straight forward, just make sure to select at least the following components
- windows/Desktop development with C++
- click on individual components and select "Windows universal CRT SDK"

### Install Cuda 9.2 Toolkit 
You can download the installer from [here](https://developer.nvidia.com/cuda-downloads). Express installation should be fine. The installer will add the necessary integration to Visual Studio.

### Create a new Cuda project
Let's test our installation by creating and running a simple Cuda project.

- In Visual Studio, select File/New/Project
- on the left side select NVIDIA/CUDA 9.2
- on the right side "CUDA 9.2 Runtime" should be selected
- give a name and path for the project and hit Ok
  
This should create a new project with a working Cuda example.

### Fix the build issue
Depending on what version of Visual Studio/Cuda Toolkit you install, the project may fail to compile with an error similar to the following:

1>c:\program files\nvidia gpu computing toolkit\cuda\v9.2\include\crt/host_config.h(133): fatal error C1189: #error:  -- unsupported Microsoft Visual Studio version! Only the versions 2012, 2013, 2015 and 2017 are supported!

I opened host_config.h and the failing check is:

```cpp
#if _MSC_VER < 1600 || _MSC_VER > 1913
```  

I looked online and [Wikipedia](https://en.wikipedia.org/wiki/Microsoft_Visual_C%2B%2B#Internal_version_numbering) has a good explanation about those versions. In particular, 1913 corresponds to Visual Studio 15.6, but in my case I installed 15.7, so we need to edit that line (after changing the permission on it) to the following:

```cpp
#if _MSC_VER < 1600 || _MSC_VER > 1914
```

Now the project compiles and runs successfully, yay!

### Optional: use Cuda helper headers

Cuda samples has a nice set of helper functions that I always like to use (vector norm, overloaded operators, ...). It's installed by default in the following folder:
	
  C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2\common\inc

To be able to use those helpers we need to add this folder to the project's include directories:

- open project settings (right click on it and select properties)
- on the left side select VC++ Directories
- on the right side edit "Include Directories" and add the path (you can just browse to it)
- make sure to make this change to both Debug and Release configurations
  
Then you can just include it as follows:

```cpp
#include <helper_math.h>
```
