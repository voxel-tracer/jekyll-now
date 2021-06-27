---
published: false
---
## Better Debugging with Rendering Events

## Problem I wanted to solve
As I was working on my rendering tool I've stumbled across a rendering bug that I couldn't explain. Basically it looked like rays were refracting inside a diffuse sphere instead of reflecting at its surface:

![rendering-bug]({{site.baseurl}}/images/render_bug.png)

## How I used to solve this problem

Usually when I find such an issue I go through the following process to identify the cause:
- reduce the number of samples per pixel until I can reproduce the issue for a single sample. This is a manual process that can take quite some time
- add a bunch of printf() inside the renderer to understand what is happening. Start with a few well placed printfs and as I get a better understanding of the issue, add more printf around what I suspect is causing the issue
- once I narrow down which part of the code is causing the issue, debug the code and look at the various variables until I understand what caused the issue to happen

## Limitations of this approach

- this is a very laborious and time consuming work that needs to be repeated every time I find a rendering bug.
- apart of the bug fixing, all the other work will be thrown away as it will otherwise clutter the rendering code making it hard to maintain
- there is no easy way to confirm I fixed the bug for all samples and not just for the one I debugged, also there is no way to ensure the bug doesn't happen again in the future.

## Rendering events to the rescue

One of the goals I had for my new project was to implement more advanced materials like subsurface scattering and microfacet model. I knew from the start I will be debugging rendering bugs and wanted to try a better approach this time.

The idea I had was to design the renderer such that it's easy to toggle debug printing without cluttering its code too much. For instance, here is the main rendering loop, inspired by [this stackoverflow post](https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering):

```cpp
color ray_color(const ray& r, shared_ptr<rnd> rng) { 
        color throughput = { 1, 1, 1 }; 
        color emitted = { 0, 0, 0 }; 
        shared_ptr<Medium> medium = {}; 
        ray curRay = r; 
        for (auto depth = 0; depth < max_depth; ++depth) { 
            hit_record rec; 
            if (!scene.world->hit(curRay, 0.001, infinity, rec, rng)) { 
                emitted += throughput * scene.background; 
                return emitted; 
            } 
            bool hitSurface = true; 
            // take current medium into account 
            if (medium) { 
                // check if there is an internal scattering 
                float distance = medium->sampleDistance(rng, rec.t); 
                throughput *= medium->transmission(distance); 
                if (distance < rec.t) { 
                    // ray scattered inside the medium 
                    hitSurface = false; 
                    curRay = ray( 
                        curRay.at(distance), 
                        medium->sampleScatterDirection(rng, curRay.direction()) 
                    ); 
                } 
            } 
            if (hitSurface) { 
                scatter_record srec; 
                color e = rec.mat_ptr->emitted(curRay, rec, rec.u, rec.v, rec.p) * throughput; 
                emitted += e; 
                if (!rec.mat_ptr->scatter(curRay, rec, srec, rng)) { 
                    return emitted; 
                } 
                // check if we entered or exited a medium 
                if (srec.is_refracted) { 
                    // assumes mediums cannot overlap 
                    if (medium) 
                        medium = nullptr; 
                    else 
                        medium = srec.medium_ptr; 
                } 
                if (srec.is_specular) { 
                    throughput *= srec.attenuation; 
                    curRay = srec.specular_ray; 
                    continue; 
                } 
                ray scattered; 
                double pdf_val; 
                if (scene.lights->objects.empty()) { 
                    // sample material directly 
                    scattered = ray(rec.p, srec.pdf_ptr->generate(rng)); 
                    pdf_val = srec.pdf_ptr->value(scattered.direction()); 
                } 
                else { 
                    // multiple importance sampling of light and material pdf 
                    auto light_ptr = make_shared<hittable_pdf>(scene.lights, rec.p); 
                    mixture_pdf mixed_pdf(light_ptr, srec.pdf_ptr); 
                    scattered = ray(rec.p, mixed_pdf.generate(rng)); 
                    pdf_val = mixed_pdf.value(scattered.direction()); 
                } 
                throughput *= srec.attenuation * 
                    rec.mat_ptr->scattering_pdf(curRay, rec, scattered) / pdf_val; 
                curRay = scattered; 
            } 
            // Russian roulette 
            if (depth > rroulette_depth) { 
                double m = max(throughput); 
                if (rng->random_double() > m) { 
                    return emitted; 
                } 
                throughput *= 1 / m; 
            } 
        } 
        // if we reach this point, we've exceeded the ray bounce limit, no more lights gathered 
        return emitted; 
    }
```

By looking at what the renderer is doing, I've identified the main rendering events that may happen for each sample and that could be used to understand what's going on with the renderer:

```
Event: base class for all rendering events
 New: new sample generated from camera
 Bounce: every time a sample bounces around
 Terminal: base class of all terminal events
  NoHitTerminal: ray didn't hit anything
  AbsorbedTerminal: surface didn't scatter the ray
  RouletteTerminal: ray terminated by Russian roulette
  MaxDepthTerminal: ray reached max depth
 CandidateHit: found a potential hit surface, may get ignored if inside a medium
 Hit: the ray hit something, next event is going to be a Scatter event
  MediumHit: ray intersected a medium particle
  SurfaceHit: ray intersected a surface
 Scatter:
  SpecularScatter: specular reflection/refration
  DiffuseScatter: diffuse reflection
  MediumScatter: medium scattering
```

To support these events without cluttering the renderer I created a callback class that I pass to the renderer:

```cpp
class callback { 
    public: 
        virtual void operator ()(event_ptr event) {} 
        virtual bool terminate() const { return false; } 
    };
```

I can publish events as simply as:

```cpp
if (cb) (*cb)(callback::Bounce::make(depth, throughput));
```

## Using events to identify buggy samples

One of the main benefits of this new approach is that if I know what I'm looking for I can write a callback to find if and where it's happening. In this case I want to identify all cases where a ray goes through the floor sphere (I can make it more generic and handle all diffuse objects, but let's start with this one). 

The callback will look like this:

```cpp
/* 
    * Checks if any ray goes through the a particular surface 
    * If it does then eventually it will intersect the same surface again from the inside (rec.front_face == false) 
    */ 
    class dbg_find_gothrough_diffuse : public callback { 
    private: 
        const bool verbose; 
        const bool stopAtFirstFound; 
        const std::string target;
        
        bool found = false; 
        std::shared_ptr<New> n; 
        unsigned long count = 0; 

    public: 
        dbg_find_gothrough_diffuse(std::string t, bool verbose, bool stopAtFirst)  
            : target(t), verbose(verbose), stopAtFirstFound(stopAtFirst) {} 
        virtual void operator ()(event_ptr e) override { 
            if (auto ne = cast<New>(e)) { 
                n = ne; 
                found = false; // only report/count one per sample 
            } else if (auto h = cast<SurfaceHit>(e)) { 
                if (!found && h->rec.obj_ptr->name == target && !h->rec.front_face) { 
                    ++count; 
                    if (verbose) { 
                        std::cerr << "\nFound a go through ray at (" << n->x << ", " << n->y << "):" << n->sampleId << std::endl; 
                    } 
                    found = true; 
                } 
            } 
        } 

        virtual bool terminate() const override { 
            return stopAtFirstFound && found; 
        } 

        unsigned long getCount() const { return count; } 
    };
```

The core of the callback is the following check:

```cpp
h->rec.obj_ptr->name == target && !h->rec.front_face
```

Basically the callback has a target object name, _floor_ in our case, that we know shouldn't allow rays to go through it. If any ray hits the object from the inside it will be flagged as buggy. The rest of the code allows us to run the callback with various levels of details.

Rendering the scene with _verbose = false, stopAtFirst = false_ prints the following:

```
Found 1511 buggy samples
```

Wow! that's a lot of buggy samples. Let 's run it again with _verbose = true, stopAtFirst = true_ and this time we get the first sample that has a bug:

```
Found a go through ray at (199, 41):0
```

At this point, the events change already paid for itself! not only I confirmed my renderer was buggy, I had already narrowed down to a specific sample to investigate.

## Visualizing the sample path

In general, being able to see the buggy sample path can help get a sense of the problem or at least give hints for areas to investigate further. I have another callback that I can use to construct a series of path segments:

```cpp
class build_segments_cb : public callback { 
  private: 
    vec3 p; 
    color c; 
    vec3 d; // direction 
  public: 
    virtual void operator ()(std::shared_ptr<Event> e) override { 
      if (auto n = cast<New>(e)) { 
        p = n->r.origin(); 
        d = n->r.direction(); 
        c = color(1, 1, 1); // new segments are white 
      } 
      else if (auto h = cast<Hit>(e)) { 
        segments.push_back({ toYocto(p), toYocto(h->p), toYocto(c) }); 
        p = h->p; 
      } 
      else if (auto sc = cast<Scatter>(e)) { 
        d = sc->d; 
        if (auto ss = cast<SpecularScatter>(e)) { 
          if (ss->is_refracted) 
            c = color(1, 1, 0); // refracted specular is yellow 
          else 
            c = color(0, 1, 1); // reflected specular is aqua 
        } 
        else if (cast<MediumScatter>(e)) { 
          c = color(1, 0, 1); // medium scatter is purple 
        } 
        else { 
          c = color(1, 0.5, 0); // diffuse is orange 
        } 
      } 
      else if (auto nh = cast<NoHitTerminal>(e)) { 
        // generate a segment that point towards the general direction of the scattered ray 
        segments.push_back({ toYocto(p), toYocto(p + d), toYocto(color(0, 0, 0)) }); 
      } 
    } 
   
    std::vector<tool::path_segment> segments; 
};
```

This callback is a bit more complex the not that much as long as you know the order of the events that are expected:

- New event is published when creating the camera ray, and is used to get the start of the first segment
- Hit event gives you the point that the ray intersected and is used to set the end of the previous segment and also as a start for the next segment.
- We use Scatter events to keep the scattered direction. We only use it to draw a small ray when there is NoHit as it represents a ray that reached the sky.
- Depending on which event we got, we color the segment differently (more on that below)

With this callback I got the following render:

![buggy-sample-visualized]({{site.baseurl}}/images/buggy-sample-1.png)

It's hard to tell what's going on from a still image, but in my tool I can move the camera around to get a better sense of what happened. Here is what this image tells me (keep in mind that the floor is not rendered to not clutter the image):

- the new ray (white) hit the floor and scatters in a diffuse manner (orange)
- it hits the first sphere and refracts inside it (yellow)
- a lot happens between the two spheres and probably the floor as well
- eventually the ray leaves the 2nd sphere (which has a medium) but is somehow still scattering inside a medium (this is a bug) and causes it to go inside the floor as it just ignores its surface.

This is great as we narrowed the issue further. Whatever is happening it has to do with how I handle mediums.

## Printing the sample's trace

Now that we know which sample to investigate, we can use another callback to print a trace of all of the sample's events. The callback is simple enough and can either just print the event names or a more verbose "digest" by delegating it to the event itself:

```cpp
class print_callback : public callback { 

private: 
  const bool verbose; 

public: 
  print_callback(bool verbose = false) : verbose(verbose) {} 
  
  virtual void operator ()(std::shared_ptr<Event> e) override { 
    if (auto b = cast<Bounce>(e)) { 
      std::cerr << b->depth << ":\n"; 
    } 
    else { 
      std::cerr << "\t"; 
      if (verbose) 
        e->digest(std::cerr); 
      else 
        std::cerr << e->name; 
      std::cerr << std::endl; 
    } 
  } 
};
```

Running this I got the following trace:

```
DebugPixel(199, 41) 
        new 
0: 
        candidate_hit(floor) 
        surface_hit(floor) 
        diffuse_scatter 
1: 
        candidate_hit(diffuse_sss_ball) 
        surface_hit(diffuse_sss_ball) 
        specular_scatter(refracted) 
2: 
        candidate_hit(diffuse_sss_ball) 
        medium_hit(dist = 0.0319336, rec.t = 0.542857) 
        medium_scatter 
3: 
        candidate_hit(diffuse_sss_ball) 
        surface_hit(diffuse_sss_ball) 
        specular_scatter(refracted) 
4: 
        candidate_hit(floor) 
        surface_hit(floor) 
        diffuse_scatter 
5: 
        candidate_hit(sss_ball) 
        surface_hit(sss_ball) 
        specular_scatter(refracted) 
6: 
        candidate_hit(sss_ball) 
        medium_hit(dist = 0.16406, rec.t = 0.720008) 
        medium_scatter 
7: 
        candidate_hit(sss_ball) 
        surface_hit(sss_ball) 
        specular_scatter(reflected) 
8: 
        candidate_hit(sss_ball) 
        medium_hit(dist = 0.358969, rec.t = 0.644089) 
        medium_scatter 
9: 
        candidate_hit(sss_ball) 
        medium_hit(dist = 0.218739, rec.t = 0.218739) 
        medium_scatter 
10: 
        candidate_hit(floor) 
        medium_hit(dist = 0.605081, rec.t = 0.605081) 
        medium_scatter
```

This is not the whole trace (the sample has a total of 36 bounces) but looking carefully through the trace we can see the following:

- at bounce 5 the ray refracted inside the sss_ball which has a medium
- bounces 6, 7, 8, and 9 describe how the ray is scattering inside the medium. CandidateHit event does report that the next closest hit is still sss_ball (which is expected as the ray is still inside the ball and my renderer doesn't yet support overlapping objects so I avoid those in my scenes)
- but at bounce 10, even though the previous bounce was a medium scatter, the floor is now closer to the ray than sss_ball, which means that the ray unexpectedly exited the medium without hitting its surface (sss_ball)
- the trace actually gives us a good hint of a possible cause as bounce 9 had both the medium scatter distance and the distance to its surface the same. This case should have been handled as a surface hit rather than a medium scatter.

Looking at the I immediately found the issue that I somehow never noticed before:

```cpp
if (medium) { 
  // check if there is an internal scattering 
  float distance = medium->sampleDistance(rng, rec.t); // << this is supposed to be double!!! 
  throughput *= medium->transmission(distance); 
  if (distance < rec.t) {
    ...handle medium scatter
  }
```

Fixing this and running _dbgfindgothroughdiffuse_ callback for the whole image, it now reports this:

```
Found 2 buggy samples
```

That's awesome, I only have two buggy samples now. Following the same approach as described above, I tracked and fixed both issues in a couple of hours (they were a bit more complex but both related to my medium logic). At this point, not only I was able to fix 3 bugs in a few hours, I also have an easy way to validate that this bug won't happen again.
