import librosa
import soundfile
import os, glob
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Extract features from a sound file; mfcc (Mel Frequency Cepstral Coefficient),
# chroma (12 Different pitch classes), and mel (Mel Spectrogram Frequency)
def extract_feature(fname, mfcc, chroma, mel):
    with soundfile.SoundFile(fname) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = np.mean(
                librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0
            )
            result = np.hstack((result, mfccs))

        # If chroma is true, get Short-Time Fourier Transform
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(
                librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0
            )
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

    return result


# Emotions in the RAVDESS dataset
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Emotions to observe
observed_emotions = ["calm", "happy", "fearful", "disgust"]


# Load the data and extract features for each sound file
def load_data(test_size=0.2, random_state=9):
    x, y = [], []
    for file in glob.glob("SpeechEmotion\\Actors\\Actor_*\\*.wav"):
        fname = os.path.basename(file)
        emotion = emotions[fname.split("-")[2]]

        # Skip if emotion is not part of observed emotions
        if emotion not in observed_emotions:
            continue

        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    return train_test_split(
        np.array(x), y, test_size=test_size, random_state=random_state
    )


# Split the datset
x_train, x_test, y_train, y_test = load_data()

# Create the model (Multi-Layer Perceptron Classifier)
model = MLPClassifier(
    alpha=0.01,
    batch_size=256,
    epsilon=1e-08,
    hidden_layer_sizes=(300,),
    learning_rate="adaptive",
    max_iter=500,
)

# Train the model and predict the results
model.fit(x_train, y_train)
prediction = model.predict(x_test)

# Print accuracy score
print("Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction) * 100))
