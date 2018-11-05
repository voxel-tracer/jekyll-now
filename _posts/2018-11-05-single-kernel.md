---
published: true
---
## Optimizing memory access by combining the kernels into a single one

In the previous post we were able to render **348M rays/s**. As we've seen, accessing global memory is slow and all our optimizations were targeting those accesses. In this post we are going to rewrite the rendering logic to reduce those accesses to a minimum.

The general rendering logic is as follows:

```
render_frame {
  generate_rays_kernel {
    stores ray to global memory
  }
  
  for each depth iteration {
    hit_world_kernel {
      loads ray
      stores hit info
    }
    scatter_kernel {
      loads ray, hit info, color, and attenuation
      stores scattered ray, color, and attenuation
    }
  }
}
```

The only reason we rely so much on global memory is to be able to communicate between the kernels, as this is the only way different kernels can exchange information. The alternative is to write a single "uber" kernel, thus reducing the usage of global memory to a minimum. The general algorithm of such a kernels is:

```
render_frame_kernel {
  generate ray
  loads color
  for each depth iteration {
    intersect ray with scene
    update ray with scattered ray
    update local color and attenuation
  }
  store color
}
```

The change has been done through a series of commits to make it easier to understand. Here is a [link to the branch](https://github.com/voxel-tracer/CudaPathTracer/tree/uber-kernel).

After making this change, the median rendering performance went up to **716.1M rays/s**. This is **~90x** the original single threaded implementation!

In the next post we are going to investigate what's limiting the performance of the kernel, if any, and how we can improve it even further.
