#!/bin/bash

video_filename="video_output.mp4"
frame_width=1280
frame_height=720
fps=60
format_cam="mjpeg"
device="/dev/video0"

# Start ffmpeg in the background
ffmpeg -y -f v4l2 -input_format "$format_cam" -video_size "${frame_width}x${frame_height}" \
       -framerate "$fps" -i "$device" -c:v libx264 -f mp4 -r "$fps" "$video_filename" &

ffmpeg_pid=$!  # Get ffmpeg process ID

echo "Recording started. Press 'ESC' to stop."

# Capture 'Esc' key and send SIGINT (Ctrl+C) to ffmpeg
while true; do
    read -rsn1 key
    if [[ $key == $'\e' ]]; then  # If Escape key is pressed
        echo "Stopping recording..."
        kill -SIGINT $ffmpeg_pid  # Send SIGINT instead of force killing
        wait $ffmpeg_pid  # Ensure ffmpeg finalizes the file properly
        echo "Recording stopped."
        exit 0
    fi
done
