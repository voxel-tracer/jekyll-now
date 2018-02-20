---
published: true
---
## How to load a png image with stb_image

Using [stb_image](https://github.com/nothings/stb/blob/master/stb_image.h) it's really easy to load a png image in memory. 

First all the needed code is in a single header file, so you can just import that one file. 

```cpp
	#define STB_IMAGE_IMPLEMENTATION
	#define STBI_ONLY_PNG
	#include "stb_image.h"
```

You only need to define _STB_IMAGE_IMPLEMENTATION_ in one of the C or C++ to create the implementation. You can optionally define _STBI_ONLY_PNG_ to reduce your code footprint.

You can them call _stbi_load_ like this:

```cpp
	int nx, ny, n;
	const int desired_channels = 3;
	unsigned char *data = stbi_load("my_picture.png", &nx, &ny, &n, desired_channels);
	if (data == NULL)
		return 1;
```

This will load the image in _data_, and populate the variables _nx_, _ny_, _n_, with respectively the width and height of the loaded image in pixels and the number of channels in the image. The passed _desired_channels_ tells _stbi_load_ how many channels the loaded data should have, in this case 3 (red, green, and blue).
Make sure you check if the returned data is not null

Once the data is loaded you can access the pixels like this:

```cpp
	for (int y = 0, idx = 0; y < ny; y++) {
		for (int x = 0; x < nx; x++, idx++) {
			unsigned char red = data[idx * desired_channels];
			unsigned char green = data[idx * desired_channels + 1];
			unsigned char blue = data[idx * desired_channels + 2];
			// ...
		}
	}

```

Once you are done with the loaded image, you should free the data using the following:

```cpp
	stbi_image_free(data);
```

That's it! Take a look at stb_image as it supported a lot of image formats. In a later post I will show how we can use SDL to display the image.
