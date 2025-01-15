import tkinter as tk
from tkinter import messagebox
import pyaudio
import wave
import numpy as np
import threading

# Global Constants
auto_record_threshold = 0.02  # Volume threshold for auto-recording

def print_device_info():
    """Print detailed information about available audio devices."""
    audio = pyaudio.PyAudio()
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        print(f"Device Index: {i}")
        print(f"Name: {info['name']}")
        print(f"Max Input Channels: {info['maxInputChannels']}")
        print(f"Max Output Channels: {info['maxOutputChannels']}")
        print(f"Default Sample Rate: {info['defaultSampleRate']}")
        print("-" * 40)
    audio.terminate()

# Call this function to inspect device capabilities
print_device_info()

# Audio Utility Functions
def detect_audio_devices():
    """Detect available audio input/output devices."""
    audio = pyaudio.PyAudio()
    devices = []

    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        device_name = info["name"]
        max_input_channels = info["maxInputChannels"]
        max_output_channels = info["maxOutputChannels"]

        if max_input_channels > 0:
            devices.append({"name": device_name, "type": "Microphone", "index": i})
        if max_output_channels > 0:
            devices.append({"name": device_name, "type": "Speaker", "index": i})

    audio.terminate()
    return devices


def record_audio(device_index, rate, event, filename="recorded_audio.wav"):
    """Record audio from the selected device."""
    audio = pyaudio.PyAudio()
    device_info = audio.get_device_info_by_index(device_index)
    channels = min(device_info["maxInputChannels"], 2)  # Use max available channels, capped at 2

    try:
        stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                            input=True, input_device_index=device_index, frames_per_buffer=1024)

        frames = []
        while event.is_set():
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        save_audio(frames, channels, rate, filename)
    except OSError as e:
        audio.terminate()
        print(f"Error: {e}")



def detect_and_record(device_index, rate, event, filename="auto_recorded_audio.wav"):
    """Automatically record when sound is detected."""
    audio = pyaudio.PyAudio()
    channels = 1
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                        input=True, input_device_index=device_index, frames_per_buffer=1024)

    frames = []
    while event.is_set():
        data = stream.read(1024, exception_on_overflow=False)
        volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean() / 32768

        if volume > auto_record_threshold:
            frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    save_audio(frames, channels, rate, filename)


def save_audio(frames, channels, rate, filename):
    """Save recorded audio to a .wav file."""
    with wave.open(filename, "wb") as wf:
        audio = pyaudio.PyAudio()
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b"".join(frames))


# GUI Class
def init_ui(app):
    """Initialize the GUI components."""
    tk.Label(app.root, text="Select a device to record:", font=("Arial", 14)).pack(pady=10)

    # Device selection
    for device in app.devices:
        tk.Radiobutton(app.root, text=f"{device['name']} ({device['type']})",
                       value=device["name"], variable=app.device_var, command=app.update_device).pack(anchor="w")

    # Sampling rate selection
    tk.Label(app.root, text="Select Sampling Rate (Hz):", font=("Arial", 12)).pack(pady=10)
    tk.Scale(app.root, variable=app.rate_var, from_=16000, to=48000, resolution=1000,
             orient="horizontal", length=400).pack(pady=10)

    # Buttons
    app.record_button = tk.Button(app.root, text="Record", state="disabled", command=app.start_recording)
    app.record_button.pack(pady=5)

    app.stop_button = tk.Button(app.root, text="Stop", state="disabled", command=app.stop_recording)
    app.stop_button.pack(pady=5)

    app.auto_record_button = tk.Button(app.root, text="Auto Record on Sound", state="disabled", command=app.auto_record)
    app.auto_record_button.pack(pady=5)

    # Status label
    app.status_label = tk.Label(app.root, text="Select a device to get started.", font=("Arial", 12))
    app.status_label.pack(pady=10)


class AudioRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder")
        self.root.geometry("600x400")

        # Recording control variables
        self.recording_event = threading.Event()
        self.auto_record_event = threading.Event()
        self.selected_device = None

        # Tkinter variables
        self.rate_var = tk.IntVar(value=16000)  # Default rate
        self.device_var = tk.StringVar(value="")

        # Device list
        self.devices = detect_audio_devices()

        # GUI components
        init_ui(self)

    def update_device(self):
        """Update the selected device."""
        device_name = self.device_var.get()
        self.selected_device = next((d for d in self.devices if d["name"] == device_name), None)
        if self.selected_device:
            self.record_button.config(state="normal")
            self.auto_record_button.config(state="normal")
            self.status_label.config(text=f"Selected device: {self.selected_device['name']}")

    def start_recording(self):
        """Start recording audio."""
        if not self.selected_device:
            messagebox.showerror("Error", "No device selected.")
            return

        self.recording_event.set()
        self.status_label.config(text="Recording... Press Stop to end.")
        self.record_button.config(state="disabled")
        self.stop_button.config(state="normal")

        threading.Thread(target=record_audio, args=(self.selected_device["index"], self.rate_var.get(),
                                                    self.recording_event), daemon=True).start()

    def stop_recording(self):
        """Stop recording audio."""
        self.recording_event.clear()
        self.auto_record_event.clear()
        self.status_label.config(text="Recording stopped.")
        self.record_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.auto_record_button.config(state="normal")

    def auto_record(self):
        """Start automatic recording based on sound detection."""
        if not self.selected_device:
            messagebox.showerror("Error", "No device selected.")
            return

        self.auto_record_event.set()
        self.status_label.config(text="Auto recording... Press Stop to end.")
        self.record_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.auto_record_button.config(state="disabled")

        threading.Thread(target=detect_and_record, args=(self.selected_device["index"], self.rate_var.get(),
                                                         self.auto_record_event), daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioRecorderApp(root)
    root.mainloop()
