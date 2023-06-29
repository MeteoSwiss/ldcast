#!/bin/bash
ffmpeg -r 1.5 -pattern_type glob -i '*.png' -c:v libx264 -vf fps=25,crop=734:584:0:0 -pix_fmt yuv420p $1
