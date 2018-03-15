from data.librispeechdata import LibriSpeechData
from models.deepspeech2 import DeepSpeech2

if __name__ == "__main__":
    folder_paths = ['../data/LibriSpeech']
    data = LibriSpeechData('mfcc', 12, 'transcription_chars', 10, 150, 100, folder_paths)

    batch = data.batch_generator(tf=True)

    model = DeepSpeech2(data, True)

    
