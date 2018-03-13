import watson_developer_cloud as wat
from os.path import join, dirname
import pyaudio
import wave


class TTS:

    ROOT_PATH = dirname(dirname(__file__))
    TEMP_FILE = join(ROOT_PATH, 'res', 'output.wav')

    @staticmethod
    def say(text):

        text_to_speech = wat.TextToSpeechV1(
            username='82c7b218-5aba-43f6-b0fc-1a81c9f529a7',
            password='RqVkPlCcbW44'
        )

        with open(TTS.TEMP_FILE,'wb') as audio_file:
            audio_file.write(
                text_to_speech.synthesize(text, accept='audio/wav',
                                          voice="en-US_AllisonVoice").content)

        chunk = 1024

        #open a wav format music
        f = wave.open(TTS.TEMP_FILE,"rb")
        #instantiate PyAudio
        p = pyaudio.PyAudio()
        #open stream
        stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                        channels = f.getnchannels(),
                        rate = f.getframerate(),
                        output = True)
        #read data
        data = f.readframes(chunk)

        #play stream
        while data:
            stream.write(data)
            data = f.readframes(chunk)

        #stop stream
        stream.stop_stream()
        stream.close()

        #close PyAudio
        p.terminate()
        return