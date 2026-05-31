# Optical Character Recognition of a 8-segment multiple display device

This project was made for my Machine Vision Systems course in TEC. It consisted in the use of a camera to recognize the digits of several 7/8 segment displays, in this case, a digital alarm clock.

There are several open code OCR libraries available that have automated and optimized this process, but to create the code from a scratch using only the knowledge adquired in the course at the moment, which was the basics of ligthing and segmentation, and logic operations on images.

The python code provided uses OpenCV and scikit-image. The program captures the image from a camera, applies preprocessing and segmentation techniques, detects individual digits, and classifies them trough a custom segment-based pattern recognition to extract the displayed time. 

In the images provided, you can see the clock used for the project and the binarization result. Also, the display separation result for an the 11:43 P.M. example. Notice that the firs separated segment is a dot, indication that the hour corresponds to the P.M. portion. Logic in the code was programmed to handle this difference, as the A.M. portions do not have this first circular segment on. 
