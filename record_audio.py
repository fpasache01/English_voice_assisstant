import sounddevice as sd
from scipy.io.wavfile import write

def list_audio_devices():
    print("\nAvailable audio devices:")
    for idx, device in enumerate(sd.query_devices()):
        print(f"{idx}: {device['name']} - Max Input Channels: {device['max_input_channels']} - Max Output Channels: {device['max_output_channels']}")

list_audio_devices()


# Set the device ID for the loopback device
LOOPBACK_DEVICE_ID = 3  # Replace with the actual ID of your loopback device

# Recording settings
SAMPLE_RATE = 16000  # CD-quality sample rate
DURATION = 10  # Duration of the recording in seconds
CHANNELS = 2  # Stereo recording

# Set the loopback device for recording
sd.default.device = LOOPBACK_DEVICE_ID

print(f"Recording system audio for {DURATION} seconds...")

# Start recording
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS)
sd.wait()  # Wait for the recording to finish

# Save the recording to a WAV file
output_filename = "system_audio_recording.wav"
write(output_filename, SAMPLE_RATE, recording)

print(f"Recording saved as {output_filename}")