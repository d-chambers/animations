#!/bin/bash
# Create the animation using ffmeg
#convert -delay 7 -quality 100 -density 300 -loop 0 stalta_single/*.png output.mp4
ffmpeg -i stalta_single/%03d.png -c:v libx264 -strict -2 -preset slow -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -f mp4 output.mp4
