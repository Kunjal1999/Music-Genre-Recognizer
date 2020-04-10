# Music-Genre-Classifier 

Does what the name suggests. Classifies songs into genres: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock.

## Dataset
fma-small dataset was used for training and testing. The dataset consists of 8,000 tracks of 30s, with 1000 tracks for each of the 8 genres. The dataset can found [here](https://github.com/mdeff/fma).

## Preprocessing
The tracks are first converted to their corresponding [Mel Spectrograms](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0) using the [librosa](https://librosa.github.io/librosa/) library in Python. The conversion may take quite some time on low powered CPUs, hence one may try only a few tracks at a time and save the results as ```bash
numpy``` arrays.

## CNN-RNN Parallel
The [CNN-RNN_parallel](https://github.com/Saif807380/Music-Genre-Classifier/blob/master/CNN-RNN_parallel.ipynb) notebook uses the compressed spectograms to build a a parallel CNN-RNN model in Keras.

## Installation
You can install the necessary python libraries post creating a virtual environment or globally by
```bash
pip install -r requirements.txt
```

## Model Weights
The best trained model weights can be found [here](https://drive.google.com/open?id=19Mzl_29lUKtbGQ7jpscIxGJ--UyJztEd).

## LICENSE
[MIT](https://choosealicense.com/licenses/mit/)
