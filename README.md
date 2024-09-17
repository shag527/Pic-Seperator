# Pic-Seperator

Faces are high dimensionality data consisting of a number of pixels. Facial recognition is a way of recognizing a human face through technology. For this, various pre-built libraries are available. 

We have used the OpenCV Haar Cascade classifier for face detection. However, resultant data which is in high dimensionality is difficult to process and cannot be visualized using simple techniques. So, we have used Principal Component Analysis (PCA) to reduce the high dimensionality of data and then fed it into the Support Vector Machine (SVM) classifier to classify the Bill Gates vs Non Bill Gates images.
