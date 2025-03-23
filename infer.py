import wave

import requests


prompt = """
What the fuck did you just fucking say about me, you little bitch? 
I'll have you know I graduated top of my class in the Navy Seals, and I've been involved in numerous secret raids on Al-Quaeda, 
"""

response = requests.post('http://localhost:8000/v1/audio/speech', json=dict(input=prompt, stream=True))

wav_file = wave.open('output.wav', "wb")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(24000)

for chunk in response.iter_content():
    wav_file.writeframes(chunk)

wav_file.close()