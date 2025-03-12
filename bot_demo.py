from paz.pipelines import DetectMiniXceptionFER
from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
import time
import pygame
from mutagen.mp3 import MP3 as mp3
from gtts import gTTS

# Initialize the facial emotion recognition pipeline and camera
pipeline = DetectMiniXceptionFER([0.1, 0.1])
camera = Camera(0)
# Start the video player
player = VideoPlayer((640, 480), pipeline, camera)
player.run()

# Define responses based on detected emotion
with open('classnames.txt') as f:
    classnames = f.readlines()
    last_classname = classnames[-1].replace('\n', '')
    print(last_classname)
    recommended_texts = ''
    if last_classname == 'neutral':
        recommended_texts = 'Hey there, just chilling?'
    elif last_classname == 'happy':
        recommended_texts = 'Whoa! Did you just win the lottery?'
    elif last_classname == 'surprise':
        recommended_texts = 'Did I surprise you?'
    elif last_classname == 'sad':
        recommended_texts = 'Cheer up! Free cookies exist somewhere!'
    elif last_classname == 'angry':
        recommended_texts = 'Whoa, whoa! Do not flip the table yet!'
    elif last_classname == 'disgust':
        recommended_texts = 'Did someone just put pineapple on pizza?'
    elif last_classname == 'fear':
        recommended_texts = 'Are you scared of me?'
    print(recommended_texts)

# Define the path to save the generated speech file
mp3_path = r"/Users/sasakitomo/Desktop/paz-master"

# Convert the response text to speech
language = 'en'
output = gTTS(text=recommended_texts, lang=language, slow=False)
mp3_file = mp3_path + "/output.mp3"
output.save(mp3_file)

# Initialize pygame mixer for audio playback
file_name = mp3_file
pygame.mixer.init()
pygame.mixer.music.load(file_name)
mp3_length = mp3(file_name).info.length
pygame.mixer.music.play(1)  # Play the MP3 file once
time.sleep(mp3_length + 0.25)  # Wait for the audio to finish playing
pygame.mixer.music.stop()  # Stop the playback