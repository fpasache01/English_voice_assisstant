#!/bin/bash

# Get the monitor device name
MONITOR_DEVICE='alsa_output.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_sofhdadsp__sink.monitor'


# Check if a monitor device was found
if [ -z "$MONITOR_DEVICE" ]; then
    echo "No monitor device found. Ensure that PulseAudio is running and a monitor source is enabled."
    exit 1
fi

echo "Detected monitor device: $MONITOR_DEVICE"

# Define the output file with a timestamp
OUTPUT_FILE="recorded_audio_$(date +%Y%m%d_%H%M%S).wav"
echo "Recording internal audio to: $OUTPUT_FILE"

# Record the audio
ffmpeg -y -f pulse -i "$MONITOR_DEVICE" "$OUTPUT_FILE"