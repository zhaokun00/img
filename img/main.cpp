#include <stdio.h>
#include<stdlib.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv; 

/*
	示例1:测试图片的读取和显示
*/
void test1();

/*
	示例2:测试从图片中读取像素值
*/
void test2();

/*
	示例3:测试用指针从数组中读取数据
*/
void test3();

int main() {

	printf("Hello Opencv\n");
	
	//test1();
	test2();
	//test3();

	return 0;
}

void test1() {

	/*	
		Opencv--waitKey()函数详解
		https://blog.csdn.net/farmwang/article/details/74170975

		图像的深度
		https://blog.csdn.net/u011484045/article/details/43573797
		https://blog.csdn.net/zxjor91/article/details/46584871
		https://blog.csdn.net/qq_29540745/article/details/52487832
	*/
	Mat src, dst;

	src = imread("D:/test.png");
	if (!src.data) {
		printf("could not load image...\n");
		return;
	}

	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", src);

	waitKey(0);
}

void test2() {

	Mat src;
	src = imread("D:/test.bmp");
	if (src.empty()) {
		printf("could not load image...\n");
		return;
	}

	namedWindow("input", CV_WINDOW_AUTOSIZE);
	imshow("input", src);

	int rows = src.rows;
	int colums = src.cols;
	int channels = src.channels();

	printf("rows = %d\n",rows);
	printf("colums = %d\n", colums);
	printf("channels = %d\n", channels);

	printf("depth = %d\n", src.depth());
	printf("type = %d\n", src.type());

	for (int row = 0; row < rows; row++) {
		for (int col = 0; col < colums; col++) {

			printf("**********************\n");
			int blue = src.at<Vec3b>(row, col)[0];
			int green = src.at<Vec3b>(row, col)[1];
			int red = src.at<Vec3b>(row, col)[2];

			printf("blue = %x\n", blue);
			printf("green = %x\n", green);
			printf("red = %x\n", red);
		}
	}

	waitKey();
}

void test3() {

	char buffer[3] = {1,2,3};

	char *p = buffer;

	printf("p[0] = %d\n",p[0]);
	printf("*p = %d\n", *p);

	system("pause");
}