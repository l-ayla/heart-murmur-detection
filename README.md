# 🫀 Heart Murmur Detection with Deep Learning

This app is a simple front end to my dissertation project which explored deep learning optimisation for heart murmur detection, it is based on the PhysioNet 2022 George B. Moody Challenge, for which the database on which it is trained can be installed 
```
wget -r -N -c -np https://physionet.org/files/circor-heart-sound/1.0.0/
```



### How it works 
1. #### Upload File
Upload a ```.wav``` file of a heart recording (minimum 9 seconds for baseline accuracy)

1. ```View Segment ```
The uploaded file is then preprocessed and converted into a 128-Mel-Feature Spectrogram to be optionally viewed
   

2. ```Analyse```
The ```pcg.h5``` classifies the file as one of three classes 
- Murmur Present
- Murmur Absent
- Unknown 
Any recording with less than 50% confidence is re-classified as unknown automatically 

### CNN Model 


  
