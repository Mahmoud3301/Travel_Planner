import speech_recognition as sr
import tempfile
import os

def transcribe_audio(audio_data, language="en-US"):
    """Convert audio to text using Google Speech Recognition"""
    recognizer = sr.Recognizer()
    
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.read())
            tmp_filename = tmp_file.name
        
        # Use the file with recognizer
        with sr.AudioFile(tmp_filename) as source:
            audio = recognizer.record(source)
        
        # Recognize speech
        text = recognizer.recognize_google(audio, language=language)
        
        # Clean up temporary file
        os.unlink(tmp_filename)
        
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error processing audio: {str(e)}"

def get_language_code(language_name):
    """Get speech recognition language code from language name"""
    language_codes = {
        "English": "en-US",
        "Arabic": "ar-SA"
    }
    return language_codes.get(language_name, "en-US")

if __name__ == "__main__":
    # Test the voice processing
    print("Voice processing module loaded successfully")
    print("Available language codes:", get_language_code("English"), get_language_code("Arabic"))
