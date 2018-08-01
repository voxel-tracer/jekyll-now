---
published: true
---
# How to write an optimized CUDA Path Tracer

I'm finally going to do it. Take all what I learned in the past year writing optimized CUDA code and apply it to improve the performance of [Peter Shirley](https://twitter.com/Peter_shirley)'s [Raytracer](https://twitter.com/Peter_shirley/status/985561344555417600). This will help me to both share what I learned with others and to pickup new optimization tricks and see how they fare on a real application.

The main motivation is to show how easy/hard it is to use CUDA to boost the performance of existing applications, to learn how to identify performance bottlenecks, and to improve them. The full source code will be available in my [github](https://github.com/voxel-tracer) (will add link to the specific repo later).

I will be using [Aras](https://twitter.com/aras_p) slightly modified implementation so we can compare the performance of the CUDA implementation to the [other implementations](http://aras-p.info/blog/2018/03/28/Daily-Pathtracer-Part-0-Intro/) Aras already did but I will only focus on the Windows implementation for now. I will probably make some simplications to it to make it easier to port to Linux as I plan to run it on a GCE instance at some point (more on that in the next post).

Disclaimer: I've been doing this for more than a year now, and I've definitely learned a lot when it comes to writing CUDA kernels, but I will also be exploring new ideas along the way that may or may not improve the performance of the raytracer. I will also be learning how to write technical blog posts, and I will be using the same structure as Aras daily pathtracer because I've enjoyed reading those posts and learned a lot from them.

# All posts in this series

- [01: Code Preparation](https://voxel-tracer.github.io/Code-Preparation/): 8M rays/s
- [02: Your first Cuda project](https://voxel-tracer.github.io/Your-First-Cuda-Project/)
- [03: Lightweight Kernel](https://voxel-tracer.github.io/lightweight-kernel/): 8.6M rays/s
- [04: Optimize Memory Transfers](https://voxel-tracer.github.io/Optimize-Memory-Transfers/): 12.3 M rays/s
- [05: Compact non active Rays](https://voxel-tracer.github.io/compact-non-active-rays/): 15.8M rays/s

# Useful resources

- Peter Shirley's [raytracing books](https://twitter.com/Peter_shirley) are a very good practical, hands down, introduction to raytracing.
- Aras's [Daily Path tracer](http://aras-p.info/blog/2018/03/28/Daily-Pathtracer-Part-0-Intro/) series, is the main inspiration behind my work, and will mainly using the implmentation from there and comparing the performance to the various implementations Aras did.
- Udacity's [Intro to Parallel Programming](https://eu.udacity.com/course/intro-to-parallel-programming--cs344) is a very good, and free, CUDA course. That's what where I learned how to write CUDA code.
