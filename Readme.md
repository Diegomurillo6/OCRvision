# Optical Character Recognition of a 8-segment multiple display device

This project was made for my Machine Vision Systems course in TEC. It consisted in the use of a camera to recognize the digits of several 7/8 segment displays, in this case, a digital alarm clock.

There are several open code OCR libraries available that have automated and optimized this process, but to create the code from a scratch using only the knowledge adquired in the course at the moment, which was the basics of ligthing and segmentation, and logic operations on images.

The python code provided uses OpenCV and scikit-image. The program captures the image from a camera, applies preprocessing and segmentation techniques, detects individual digits, and classifies them trough a custoom segment-based pattern recognition to extract the displayed time. 
