#!/bin/bash
# Create the animation using ffmeg
convert -delay 7 -quality 100 -density 300 -loop 0 cofilt/*.png output.mp4


#ffmpeg -i cofilt/%03d.png output.mp4
