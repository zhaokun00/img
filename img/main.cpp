#include <stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace cv; int main() {

	printf("Hello Opencv\n");

	Mat src, dst;
	src = imread("D:/test.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);

	waitKey(0);
	return 0;
}