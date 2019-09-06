#!/bin/bash
# Create the animation using ffmeg
convert -delay 7 -quality 100 -density 300 -loop 0 stalta_single/*.png output.mp4