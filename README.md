# wallpp

An application that converts a set of images into collaged wallpapers of a set ratio.

![collage demonstration](https://github.com/naelstrof/wallpaper-post-processor/assets/1131571/0ddea205-f8cb-439f-ba5b-b8a452b60ff1)

## Features

- Uses AI upscaling to keep images in a similar resolution range.
- Does appropriate delta updates, regenerate wallpapers from a set of source images with minimal work. (Delete and add new source images!)
- Uses magic numbers to read resolutions, scan a set of 4000 images in 6 seconds (on an SSD) for future processing.
- Reads any images supported by OpenCV.
- Outputs only JPGs, with limits to ensure you aren't overfilling your harddrives.

## Usage

This application is entirely CLI powered.

Normally you'd execute it with the following parameters:

```bash
wallpp --input-path=/mnt/d/Pictures/Wallpapers/ --output-path=/mnt/d/Pictures/WallpapersOutput/ --upscaler-path=~/wallpaper-post-processor/
```

Where `--input-path` is the path to your set of images you want to collage, and `--output-path` is the location where you want your collages placed. BEWARE that images within output-path are mutable and will be removed and regenerated when necessary.

`--upscaler-path` is the folder containing both the `FSRCNN_x2.pb` and the `FSRCNN_x4.pb` AI upscaling files.

Files within `--input-path` are considered immutable and won't be altered.

If you're on windows, you should consider making a shortcut to take care of it.

## Installation

Simply download an executable from Releases, otherwise use cargo to install and compile. Opencv is rough to get on Windows, check [here](https://github.com/twistedfall/opencv-rust) for instructions on how to get it.
