# 🫀 Heart Murmur Detection with Deep Learning

This streamlit applet is a simple front end to a short project exploring deep learning optimisation for heart murmur detection, it is based on the [2022 George B. Moody PhysioNet Challenge](https://moody-challenge.physionet.org/2022/), for which the database on which it is trained can be installed 
```
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.3/
```
This applet is available to use [here](murmur-detection.streamlit.app)


### How it works 
1. #### Upload File
Upload a ```.wav``` file of a heart recording (minimum 9 seconds for baseline accuracy)

2. #### View Segment 
The uploaded file is then preprocessed and converted into a 128-Mel-Feature Spectrogram to be optionally viewed
   

2. #### Analyse 
The ```pcg.h5``` classifies the file as one of three classes 
- Murmur Present
- Murmur Absent
- Unknown 

###### Any recording with less than 50% confidence is re-classified as unknown automatically 

### CNN Model 

- model size 
- epochs 
- learning rate 
- diagram 
- architecture 



  
