from __future__ import print_function
import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np
from matplotlib import cm


fn_abnormal = '/Users/mohamedfawzy/Workspace/panotti/examples/physionet2016/training-a/a0059.wav'
fn_normal = '/Users/mohamedfawzy/Workspace/panotti/examples/physionet2016/training-a/a0050.wav'


# Generate mfccs from a time series

y, sr = librosa.load(fn_normal)
librosa.feature.mfcc(y=y, sr=sr)
# array([[ -5.229e+02,  -4.944e+02, ...,  -5.229e+02,  -5.229e+02],
# [  7.105e-15,   3.787e+01, ...,  -7.105e-15,  -7.105e-15],
# ...,
# [  1.066e-14,  -7.500e+00, ...,   1.421e-14,   1.421e-14],
# [  3.109e-14,  -5.058e+00, ...,   2.931e-14,   2.931e-14]])

# Use a pre-computed log-power Mel spectrogram

S = librosa.feature.melspectrogram(y=y, sr=sr)

pylab.figure(figsize=(10, 4))
pylab.axis('off')
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
pylab.savefig('test-melspec.jpg', bbox_inches=None, pad_inches=0)
pylab.close()


#librosa.feature.mfcc(S=librosa.power_to_db(S))
# array([[ -5.207e+02,  -4.898e+02, ...,  -5.207e+02,  -5.207e+02],
# [ -2.576e-14,   4.054e+01, ...,  -3.997e-14,  -3.997e-14],
# ...,
# [  7.105e-15,  -3.534e+00, ...,   0.000e+00,   0.000e+00],
# [  3.020e-14,  -2.613e+00, ...,   3.553e-14,   3.553e-14]])

# Get more components

mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S, ref=np.max))

# Visualize the MFCC series
#librosa.display.specshow(mfccs, cmap='Set1',x_axis='time')
# Plotting the spectrogram and save as JPG without axes (just the image)
pylab.figure(figsize=(10, 4))
pylab.axis('off')
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
librosa.display.specshow(mfccs, cmap='Set1')
pylab.savefig('test.jpg', bbox_inches=None, pad_inches=0)
pylab.close()