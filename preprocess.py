import os
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# def getTrackIds(directory):
#     trackIds = list()
#     for root,dname,files in os.walk(directory):
#         if dname == []:
#             trackIds.extend(file[:-4] for file in files)
#     return trackIds

# returns path to audio file with given id
def getTrack(directory,id):
    trackId = '{:06d}'.format(id)
    return os.path.join(directory,trackId[:3],trackId + '.mp3')

# loads the track data csv file
def loadMetadata(directory):
    tracksMetadata = pd.read_csv(directory,index_col=0, header=[0, 1])
    return tracksMetadata

def createMelSpectrogram(trackId,genre):
    song = getTrack('../fma_small',trackId)
    y, sr = librosa.load(song)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 1024)
    spectrogram = librosa.power_to_db(spectrogram,ref = np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title(str(genre))
    plt.show()



# preprocessing...
df = loadMetadata('../fma_metadata/tracks.csv')
keepCols = [('set', 'split'),('set', 'subset'),('track', 'genre_top')]
df = df[keepCols]
df = df[df[('set','subset')]=='small']
df['track_id'] = df.index

print("Shape of the data:")
print(df.shape)
print()
print()
print("Genre wise distribution:")
print(df[('track','genre_top')].value_counts())
print()
print()
print("Train Cross Validation Test Split: ")
print(df[('set', 'split')].value_counts())

groupByGenre = df.groupby(('track','genre_top')).first().reset_index()

print()
print("Creating Spectograms for each genre: ")

for index, row in groupByGenre.iterrows():
    trackId = int(row['track_id'])
    genre = row[('track', 'genre_top')]
    createMelSpectrogram(trackId, genre)
