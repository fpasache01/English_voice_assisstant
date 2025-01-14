import json
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import openai
from speechbrain.pretrained import SpeakerRecognition
import keyboard
import os.path

# Set OpenAI API key
api_key = "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"  # Replace with your actual API key

openai.api_key = api_key

# File paths and constants
CHAT_HISTORY_FILE = "chat_history.json"
CHAT_LOG = "chat_log.json"
YOUR_VOICE_PROFILE = "your_voice.wav"
TEMP_AUDIO_PATH = "temp_audio.wav"
SAMPLE_RATE = 16000
DURATION = 5  # Recording duration in seconds
# Constants
CHANNELS = 1  # Mono audio
FRAME_DURATION = 0.1  # Duration of each audio frame in seconds
ENERGY_THRESHOLD = 500  # Threshold to detect voice activity
MAX_SILENCE_FRAMES = 10  # Number of silent frames before stopping

# Initialize Speaker Recognition model
speaker_recognition = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb" )

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def record_audio_auto_detect(samplerate=SAMPLE_RATE, channels=CHANNELS, frame_duration=FRAME_DURATION):
    frame_size = int(samplerate * frame_duration)  # Number of samples per frame
    audio_buffer = []
    silence_count = 0

    print("Listening for voice...")

    # Stream audio from the microphone
    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16') as stream:
        while True:
            # Read audio frame
            audio = stream.read(frame_size)[0].flatten()

            # Calculate the energy level (sum of squares)
            energy = np.sum(audio.astype(np.int32) ** 2) / len(audio)

            if energy > ENERGY_THRESHOLD:
                # Voice detected; reset silence counter and store audio
                silence_count = 0
                audio_buffer.append(audio)
                #print("Voice detected.")
            else:
                # Silence detected; increment silence counter
                silence_count += 1
                #print("Silence detected.")

            # Stop recording if silence persists
            if silence_count > MAX_SILENCE_FRAMES and audio_buffer:
                #print("Recording finished.")
                break

    # Combine recorded frames into a single NumPy array
    recorded_audio = np.concatenate(audio_buffer, axis=0)
    return recorded_audio

def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    return audio

def save_audio(audio, filepath, samplerate=SAMPLE_RATE):
    write(filepath, samplerate, audio)

def verify_speaker(audio_path, reference_path):
    score, _ = speaker_recognition.verify_files(audio_path, reference_path)
    return score

def save_chat_history(chat_history, file_path):
    with open(file_path, "w") as file:
        json.dump(chat_history, file, indent=4)

def load_chat_history(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_gpt(chat_history, prompt):
    chat_history.append({"role": "user", "content": prompt})

    try:
        chat_completion = openai.chat.completions.create( messages=chat_history, model="gpt-4o" )
        #print(completion.choices[0].message)
        assistant_message = chat_completion.choices[0].message.content
        chat_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        return f"An error occurred: {e}"

def transcribe_audio():
    model = whisper.load_model("tiny")
    chat_history = load_chat_history(CHAT_HISTORY_FILE)
    #chat_log = load_chat_history(CHAT_LOG)

    print("Listening... Speak naturally.")
    try:
        while True:
            print("\nRecording... Speak now.")

            audio = record_audio_auto_detect()
            save_audio(audio, TEMP_AUDIO_PATH)

            result = model.transcribe(TEMP_AUDIO_PATH, language="en")
            text = result["text"]
           
            if text == "":              
                continue  

            similarity_score = verify_speaker(TEMP_AUDIO_PATH, YOUR_VOICE_PROFILE)
            print("similarity ")
            print(similarity_score)
            if similarity_score >= 0.65:
                print(f"{bcolors.FAIL} your voice has been detected {bcolors.ENDC}")
                continue 

            print(f"{bcolors.OKCYAN}Transcription: {text}{bcolors.ENDC}")
            #response = chat_with_gpt(chat_history, f"(in summary tell me) {text}")
            #print(f"{bcolors.OKGREEN}GPT Suggestion: {response}{bcolors.ENDC}")

    except KeyboardInterrupt:
        print("\nProgram stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    check_file = os.path.isfile(YOUR_VOICE_PROFILE)
    if(check_file):
        print("voice profile detected")
    else:
        print("Press any key to start voice detection...")
        keyboard.read_event()  # Waits for any key press
        print("Speak something for five seconds...")
        # Record and save your voice
        audio = record_audio(duration=10)  
        save_audio(audio, YOUR_VOICE_PROFILE)
        print("Your voice profile is saved.")

    # Countdown before starting transcription
    for i in range(5, 0, -1):
        print(i)

    transcribe_audio()

if __name__ == "__main__":
    main()
