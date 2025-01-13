import json
import whisper
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import openai
import string

api_key = "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"  # Replace with your actual API key
openai.api_key = api_key
chat_histories = {}

CHAT_HISTORY_FILE = "chat_history.json"
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

def save_chat_history(chat_history, file_path):
    """Save chat history to a file."""
    with open(file_path, "w") as file:
        json.dump(chat_history, file, indent=4)

def load_chat_history(file_path):
    """Load chat history from a file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return [{"role": "system", "content": "You are a helpful assistant."}]
    
def chat_with_gpt(chat_history,prompt):

    chat_history.append({"role": "user", "content": prompt})

    try:
        chat_completion = openai.chat.completions.create(
                messages=chat_history,
                model="gpt-4o",
        )           
        chat_history.append({"role": "assistant", "content": chat_completion.choices[0].message.content})
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        return f"An error occurred: {e}"

def transcribe_audio():
    """Capture real-time audio from the microphone and transcribe it using Whisper."""
    model = whisper.load_model("tiny")  # Load Whisper model
    samplerate = 16000  # Whisper requires 16 kHz audio
    print("Listening... Speak naturally.")
    chat_history = load_chat_history(CHAT_HISTORY_FILE)

    try:
        while True:
            print("\nRecording... Speak now.")

            # Record audio for a short duration
            duration = 5  # Seconds
            audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
            sd.wait()  # Wait until recording is finished
            
            # Save audio to a temporary file
            temp_audio_path = "temp_audio.wav"
            write(temp_audio_path, samplerate, audio)

            # Transcribe with Whisper
            result = model.transcribe(temp_audio_path, language="en")
            text = result["text"]
            print(text)
            if(text != ""):
                print(f"{bcolors.WARNING} Transcription: {text} , {bcolors.ENDC}")
                response = chat_with_gpt(chat_history, text)
                save_chat_history(chat_history, CHAT_HISTORY_FILE)
                print(f"{bcolors.OKGREEN} gpt suggestion: {response} , {bcolors.ENDC}")
        
    except KeyboardInterrupt:
        print("\nProgram stopped.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":    
    #chat_history = load_chat_history(CHAT_HISTORY_FILE)
    #print("Welcome to ChatGPT! Type 'exit' to quit.")

    #while True:
    #    user_input = input("You: ")
    #    if user_input.lower() == "exit":
    #        print("Saving chat history and exiting...")      
    #        print("Chat history saved.")
    #        break

        # Chat with GPT and get a response
    #    response = chat_with_gpt(chat_history, user_input)
    #    save_chat_history(chat_history, CHAT_HISTORY_FILE)
    #    print(f"ChatGPT: {response}")
    transcribe_audio()
