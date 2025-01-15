import pyaudio
import wave
import numpy as np
import threading


def detect_audio_devices_with_types():
    audio = pyaudio.PyAudio()
    devices = []

    for i in range(audio.get_device_count()):
        device_info = audio.get_device_info_by_index(i)
        device_name = device_info['name']
        max_input_channels = device_info['maxInputChannels']
        max_output_channels = device_info['maxOutputChannels']

        # Classify the device type based on its channels
        if max_input_channels > 0:
            devices.append({"name": device_name, "type": "Microphone", "index": i})
        if max_output_channels > 0:
            devices.append({"name": device_name, "type": "Speaker", "index": i})

    audio.terminate()
    return devices


def record_audio(device_index, rate, on_start, on_stop):
    print("rate")
    print(rate)
    def record():
        audio = pyaudio.PyAudio()
        device_info = audio.get_device_info_by_index(device_index)
        channels = min(device_info.get("maxInputChannels", 1), 2)  # Use the device's maximum channels, capped at 2

        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024)

        frames = []
        on_start()
        while on_stop.is_set():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("recorded_audio.wav", "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    threading.Thread(target=record, daemon=True).start()


def record_system_audio(device_index, rate, on_start, on_stop):
    def record():
        audio = pyaudio.PyAudio()
        device_info = audio.get_device_info_by_index(device_index)
        channels = min(device_info.get("maxOutputChannels", 2), 2)  # Use the device's maximum channels, capped at 2

        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024)

        frames = []
        on_start()
        while on_stop.is_set():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("recorded_system_audio.wav", "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    threading.Thread(target=record, daemon=True).start()
def record_audio(device_index, rate, on_start, on_stop):
    def record():
        audio = pyaudio.PyAudio()
        device_info = audio.get_device_info_by_index(device_index)
        channels = min(device_info.get("maxInputChannels", 1), 2)  # Use the device's maximum channels, capped at 2

        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024)

        frames = []
        on_start()
        while on_stop.is_set():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("recorded_audio.wav", "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    threading.Thread(target=record, daemon=True).start()


def record_system_audio(device_index, rate, on_start, on_stop):
    def record():
        audio = pyaudio.PyAudio()
        device_info = audio.get_device_info_by_index(device_index)
        channels = min(device_info.get("maxOutputChannels", 2), 2)  # Use the device's maximum channels, capped at 2

        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=rate,
                            input=True,
                            input_device_index=device_index,
                            frames_per_buffer=1024)

        frames = []
        on_start()
        while on_stop.is_set():
            try:
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        with wave.open("recorded_system_audio.wav", "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))

    threading.Thread(target=record, daemon=True).start()
