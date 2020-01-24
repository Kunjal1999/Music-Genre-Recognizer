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

# creates spectrogram
def createMelSpectrogram(trackId):
    song = getTrack('../fma_small',trackId)
    y, sr = librosa.load(song)
    spectrogram = librosa.feature.melspectrogram(y = y, sr = sr, n_fft = 2048, hop_length = 1024)
    spectrogram = librosa.power_to_db(spectrogram,ref = np.max)
    return spectrogram.T

# displays spectrogram
def plotSpectrogram(trackId):
    spectrogram = createMelSpectrogram(trackId)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

# creates arrays of spectrogram pixel intensities
def createArray(df):
    genres = []
    X_spect = np.empty((0, 640, 128))
    count = 0
    #Code skips records in case of errors
    for index, row in df.iterrows():
        try:
            count += 1
            track_id = int(row['track_id'])
            genre = str(row[('track', 'genre_top')])
            spect = createMelSpectrogram(track_id)
            # Normalize for small shape differences
            spect = spect[:640, :]
            X_spect = np.append(X_spect, [spect], axis=0)
            genres.append(dict_genres[genre])
            if count % 100 == 0:
                print("Currently processing: ", count)
        except:
            print("Couldn't process: ", count)
            continue
    y_arr = np.array(genres)
    return X_spect, y_arr

# splits dataframe into smaller parts for easy spectrogram computation
def splitDataFrameIntoSmaller(df, chunkSize = 1600): 
    listOfDf = list()
    numberChunks = len(df) // chunkSize + 1
    for i in range(numberChunks):
        listOfDf.append(df[i*chunkSize:(i+1)*chunkSize])
    return listOfDf

# shuffles the training and validation set data
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# preprocessing...
df = loadMetadata('../fma_metadata/tracks.csv')
keepCols = [('set', 'split'),('set', 'subset'),('track', 'genre_top')]
df = df[keepCols]
df = df[df[('set','subset')]=='small']
df['track_id'] = df.index


dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4, 'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8}

# train_validation_test split
df_train = df[df[('set', 'split')]=='training']
df_valid = df[df[('set', 'split')]=='validation']
df_test = df[df[('set', 'split')]=='test']

# splitting training dataset into smaller parts
listDf = splitDataFrameIntoSmaller(df_train)
df1_train = listDf[0]
df2_train = listDf[1]
df3_train = listDf[2]
df4_train = listDf[3]

# converting test audio to spectrograms
X_test, y_test = createArray(df_test)
np.savez('test_arr', X_test, y_test)


# converting training audio to spectrograms
X_train1, y_train1 = createArray(df1_train)
np.savez('train1_arr', X_train1, y_train1)
X_train2, y_train2 = createArray(df2_train)
np.savez('train2_arr', X_train2, y_train2)
X_train3, y_train3 = createArray(df3_train)
np.savez('train3_arr', X_train3, y_train3)
X_train4, y_train4 = createArray(df4_train)
np.savez('train4_arr', X_train4, y_train4)

# loading individual training sets and concatenating
npzfile = np.load('train1_arr.npz')
X_train1 = npzfile['arr_0']
y_train1 = npzfile['arr_1']
npzfile = np.load('train2_arr.npz')
X_train2 = npzfile['arr_0']
y_train2 = npzfile['arr_1']
npzfile = np.load('train3_arr.npz')
X_train3 = npzfile['arr_0']
y_train3 = npzfile['arr_1']
npzfile = np.load('train4_arr.npz')
X_train4 = npzfile['arr_0']
y_train4 = npzfile['arr_1']
X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4), axis = 0)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4), axis = 0)

# converting validation audio to spectrograms
X_valid, y_valid = createArray(df_valid)

# Convert y data to scale 0-7 from 1-8
y_train = y_train -1
y_valid = y_valid -1

# Convert the scale of training and validation data
X_train_raw = librosa.core.db_to_power(X_train, ref=1.0)
X_train_log = np.log(X_train_raw)
X_valid_raw = librosa.core.db_to_power(X_valid, ref=1.0)
X_valid_log = np.log(X_valid_raw)


X_train, y_train = unison_shuffled_copies(X_train_log, y_train)
np.savez('shuffled_train', X_train, y_train)

X_valid, y_valid = unison_shuffled_copies(X_valid_log, y_valid)

np.savez('shuffled_valid', X_valid, y_valid)

# print("Shape of the data:")
# print(df.shape)
# print()
# print()
# print("Genre wise distribution:")
# print(df[('track','genre_top')].value_counts())
# print()
# print()
# print("Train Cross Validation Test Split: ")
# print(df[('set', 'split')].value_counts())

# groupByGenre = df.groupby(('track','genre_top')).first().reset_index()

# print()
# print("Creating Spectograms for each genre: ")

# for index, row in groupByGenre.iterrows():
#     trackId = int(row['track_id'])
#     genre = row[('track', 'genre_top')]
#     createMelSpectrogram(trackId, genre)
