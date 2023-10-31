# RoadLane-TrafficSignDetection

1. Perspective transform (bird-eyes view)
2. Preprocessing and Thresholding
   1. Histogram Equalization
   2. Change color space to HLS to get L channel
   3. Gaussian Filter
   4. Binarize Image
3. Sliding window fit
   1. Starting Points: Once the image mask with the lane pixels is produced we can assume two big concentrations of pixels will be present were the lines are. If we produce a graph with the sum of all vertical white pixels in each horizontal pixel we will get two peaks where the lane lines should be. The starting points to start searching for lines are found in these peaks so we can get the horizontal pixel that has the maximum amount of white pixels in each half on the image and center the window to search for the line there since it's the most likely place to contain the line.
4. Polynomial fit
5. Draw detected lines
6. Perspective transform (back to normal view)
