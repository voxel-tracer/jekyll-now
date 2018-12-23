---
published: false
---
## Improving performance by increasing warp execution efficiency

In our previous post we got the render performance to **872M rays/s** which is **109x** more than the original single threaded implementation. Using the Visual profiler we can see the following issues:
- global mem load efficiency (52.5%): indicates how well the application's global loads are using device memory bandwidth. With the uber kernel most global loads are when we read the spheres and their materials and indeed it doesn't take advantage of any memory coalescing as we use constant memory and we expect the data to be cached. Maybe worth investigating further later.
- low warp execution efficiency (47%): is the average percentage of active threads in each executed warp. This is concerning as it means that we are only using half the gpu compute power on average!!!

We will look into the global memory load efficiency later, but for now let's focus on the low warp execution efficiency. As always, following the Visual Profiler guided analysis quickly points us into the right direction. _Compute Analysis_ actually gives us the exact lines where most of the thread execution divergence occurs.
Divergence happens in 3 main areas of the kernel:

A. [hitSphere()](https://github.com/voxel-tracer/CudaPathTracer/blob/uber-kernel/Cpp/Cuda/CudaRender.cu#L109) where the execution differs depending on the value of the discriminant.

B. [ScatterNoLightSampling()](https://github.com/voxel-tracer/CudaPathTracer/blob/uber-kernel/Cpp/Cuda/CudaRender.cu#L234) where the execution differs depending on which material has been intersected.

C. [RenderFrameKernel()](https://github.com/voxel-tracer/CudaPathTracer/blob/uber-kernel/Cpp/Cuda/CudaRender.cu#L350) in the main rendering loop where the number of iterations changes depending on how many bounces a ray does.

As a first step let's experiment by changing the scene a little bit to remove the divergence and see how much performance we can get out of it.
- for **A**, I'm not sure yet how to do it so we'll skip it for now
- for **B**, we can use a single material type for all spheres
- for **C**, we can use a very small max_depth so that all threads will have the same number of iterations.
Note that we are not changing the kernel code so we keep the same number of registers/occupancy.

## Experiment 1: reduce max_depth to 1

Setting max_depth to 1 removes all divergence related to **C**, we still need to go through **A** and **B** yet the profiler suggests the remaining divergence is no longer source of much bottleneck. Performance goes up to **1.2G rays/s**

## Experiment 2: use the same material type for all spheres

This will get rid of most divergence except in the main rendering loop. Performance is **940.0M rays/s**. In the profiler we still see a low warp execution efficiency of **47.7%**. _Compute Analysis_ shows that most divergence is when threads don't hit anything and are done thus they remain idle while the remaining threads in the same warp are still bouncing around.

## Rethinking the main kernel loop

So it looks like the main loop divergence is by far the biggest source of thread divergence. We can think of the main rendering loop, per thread, as something like this:

```
for (uint depth = 0; depth < maxDepth; depth++)
{
  if (ray_done)
    break;
  hit_id = hit_world()
  if (hit_id > 0)                 < 44% divergence
    ray_done = ! scatter()
  else
  {
    compute sky color
    ray_done = true
  }
}
```

Profiler shows that we have **44%** divergence when we check if the ray hit anything in the scene. Which means that **44%** of the threads won't hit anything while others do, thus those threads will become idle.

Can we get more details about how many threads are going idle at each depth iteration, ideally taking warps into account ? Actually we can, as we already collect max depth reached by each thread and we copy it to the cpu, we can just store that information per frame and do some analysis on it. I hacked the renderer to printout the max depth reached by each thread before going idle, then using [Gnu Octave](https://www.gnu.org/software/octave/) I wrote a quick script to analyse those resuts:

```
load depths.txt
spp = 32;
printf ("depth\tactive threads\twarps done\twarps full\twarps divergent\n")
for d = 1:10
 DS = depths>d;
 D = reshape (DS, spp, []);
 D = sum(D);
 S = 100/size(D,2);
 printf ("%5d\t%13d%%\t%9d%%\t%9d%%\t%14d%%\n", d, round(sum(DS) * 100 / size(DS, 2)), round(sum(D==0)*S), round(sum(D==spp)*S), round(sum(D>0 & D<spp)*S))
end
```

Here are the results for one frame at 32 samples per pixel:
```
depth   active threads  warps done      warps full      warps divergent
    1              85%         14%             83%                   4%
    2              25%         39%              7%                  53%
    3              13%         46%              1%                  53%
    4               4%         60%              0%                  40%
    5               2%         70%              0%                  30%
    6               1%         80%              0%                  20%
    7               1%         86%              0%                  14%
    8               0%         91%              0%                   9%
    9               0%         93%              0%                   7%
   10               0%        100%              0%                   0%
```
We can see that just after 2 depth iterations only **25%** of the threads are still active yet **53%** of the warps are divergent and only **7%** are fully active (warps that have all of their threads running).

So it does look like a lot of the divergence is caused by rays terminating because they didn't hit anything. One way to solve this is to allow the threads that finish early to process new samples up to a total defined number of depth iterations (10 in this case). As we already saw, after just 2 depth iterations **75%** of the threads are done thus can still process 8 more iterations.
Here is a [link to the change](https://github.com/voxel-tracer/CudaPathTracer/commit/67d3ececcf7df8ea8f61d24bdef792cc5f4972fc). With this change, performance went up from **872M rays/s** to **+1.2G rays/s** which is more than **152x** speed increase compared to the original single threaded implementation. Also the performance is much stable now across runs and doesn't to be affected much by how many frames we run, ie I see similar performance for 100 or 1K frames.

Looking at the Visual profiler its interesting to see warp efficiency going up from **49.5%** to **65.3%** so we still have a lot of divergence in the code. As an exercise, I modified the scene to only use Lambert material to get rid of all divergence related to material handling. Performance is **1.4G rays/s** and inactive threads went down from **40%** to **30%** so we still have divergence even with a single material.
