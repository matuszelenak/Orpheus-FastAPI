import wave

import requests


with open('copypasta', 'r') as f:
    prompt = f.read()

response = requests.post('http://localhost:4242/v1/audio/speech', json=dict(input=prompt, stream=True), stream=True)

wav_file = wave.open('output.wav', "wb")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(24000)

for chunk in response.iter_content(chunk_size=4096):
    print(len(chunk))
    wav_file.writeframes(chunk)

print('Done')
wav_file.close()