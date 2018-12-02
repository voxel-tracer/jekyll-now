---
published: true
---
Tuning the renderer for up to 100x speedup

In the previous posts, and while I was collecting the performance of my renderer, I quickly realized that depending on which settings I used the performance could be quite different. I used to manually run the renderer 10 times and compute the median performance. Now that I've arrived at a milestone in my renderer performance and it was clear that the next improvement will require a lot more work on the rendering algorithm, I wanted to spend some time to figure out the best performance I could get out of my current implementation.

## Auto tuning
I started by updating the renderer to do 10 consecutive rendering passes then compute the median rendering performance. Then I identified attributes that seem to affect performance are:
- number of rendered frames: do we have any kind of per-frame overhead that would cause the performance to drop down for large frames ?
- number of samples per pixel: the more samples we use the more work is generated for the GPU. How does it effect performance ?
- number of threads per block: choosing the right block size can help optimize device occupancy.

The auto-tuning algorithm is simple, it gets as an input a list of values for each of the identified attributes and goes through those lists systematically trying all combinations and collecting the median rendering performance for each combination. Here is a link to [the code change](https://github.com/voxel-tracer/CudaPathTracer/commit/19e10d4773b3d67fc8cbf2bf8e2e32b24e66c29f)

Here are a few interesting observations from all those runs:
- performance goes up as frames go up, no matter what combination we use. Because we don't copy any data between the host and device until the end of the rendering, there is minimal overhead of running the kernels over and over again. Actually, performance seem to improve a bit when we run for longer periods as we may be paying the same overhead price that gets amortized the longer we run.

- performance goes up as number of samples per pixel go up, but we eventually run out of memory at 64 samples. The more samples we use the more work is submitted to the device, which helps saturate the Cuda cores and hide memory and instruction latency.

- On my GTX 1050 there seem to be a sweet spot around 128-160 threads per block.

Taking all these into account, I started a **1000 frames** rendering with **32 samples per pixels** and **160 threads per block**, this yielded the best performance ever at **872.0M rays/s**. This is more than 100x faster than the single threaded implementation on my laptop just by tweaking a few parameters without making any change to the implementation at all.
