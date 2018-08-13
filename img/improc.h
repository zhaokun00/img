#ifndef _OMPROC_H_

	#define _OMPROC_H_

	#include <stdio.h>
	#include<stdlib.h>
	#include <iostream>
	#include <math.h>
	#include <opencv2/opencv.hpp>

	/*
		示例1:测试图片的读取和显示
	*/
	void improcTest1() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/01.png";

		/*
			原型:imread( const String& filename, int flags = IMREAD_COLOR );

			作用:
				加载图像
			参数:
				filename:图像文件名称
				flags:加载的图像时什么类型,支持常见的3个参数值

				IMREAD_UNCHANGED            = -1, //表示加载原图,不做任何改变
				IMREAD_GRAYSCALE            = 0,  //表示把原图作为灰度图像加载进来
				IMREAD_COLOR                = 1,  //表示把原图作为RGB图像加载进来

		*/

		//src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);
		//src = cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

#if 0
		if (!src.data) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}
#endif
		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		/*
			原型: void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
			
			作用:
				创建一个OpenCV窗口,它是由OpenCv自动创建与释放,无序人为销毁它
			参数:
				winname:窗口名字
				flags:窗口类型
				CV_WINDOW_AUTOSIZE:自动根据图像大小,显示窗口大小,不能人为改变窗口大小
				CV_WINDOW_NORMAL:允许修改窗口大小
		*/

		std::string winName = "input image";
		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);

		/*
			原型:void imshow(const String& winname, InputArray mat)

			作用:显示图像
			参数:
				winname:窗口名字
				mat:Mat对象
		*/
		cv::imshow("input image", src);

		/*
			原型:int waitKey(int delay = 0);
			
			waitKey()这个函数是在一个给定的时间内(单位ms)等待用户按键触发,如果在设置的时间内没有按键按下,则结束
			waitKey(0)表示程序会无限制的等待用户的按键事件,若发生按键则结束,否则一直等待
		*/
		cv::waitKey(0);
	}

	/*
		示例2:测试色彩空间的转换和保存图片
	*/
	void improcTest2() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/01.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "out image";

		cv::Mat dst;

		/*
			原型:cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 )

			作用:把图像从一个色彩空间转化到另外一个色彩空间
			参数:
				src:源图像
				dst:目标图像
				code:转换的色彩空间
		*/
		cv::cvtColor(src,dst,CV_BGR2GRAY);
		//cv::cvtColor(src, dest, CV_BGR2HLS);

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, dst);

		std::string outPath = "E:/study/ML/img/pic/improc/01_out.png";
		/*
			原型:bool imwrite( const String& filename, InputArray img,const std::vector<int>& params = std::vector<int>());
			作用:
				保存图像
			参数:
				filename:文件名
				img:Mat对象
			只有8位、16位的PNG、jpg、tiff文件格式而且是单通道或者三通道的BRG的图像才可以通过这种方式保存
		*/
		cv::imwrite(outPath,dst);

		cv::waitKey(0);
	}

	/*
		示例3:测试读取图片的元素

		读一个GRAY像素点的像素值（CV_8UC1）
		Scalar intensity = img.at<uchar>(y, x);
		或者 Scalar intensity = img.at<uchar>(Point(x, y));

		读一个RGB像素点的像素值
		Vec3f intensity = img.at<Vec3f>(y, x);
		float blue = intensity.val[0];
		float green = intensity.val[1];
		float red = intensity.val[2];

		Vec3b对应三通道的顺序是blue、green、red的uchar类型数据。
		Vec3f对应三通道的float类型数据
		把CV_8UC1转换到CV32F1实现如下：
		src.convertTo(dst, CV_32F);
	*/
	void improcTest3() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/02_2.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";

		int rows = src.rows; //获取图片行数
		int cloumns = src.cols; //获取图片列数
		int channels = src.channels(); //获取图片通道数

		printf("rows = %d,cloumns = %d,channels = %d\n",rows,cloumns,channels);

#if 0
		//对于RGB的图像,每一列存放了3字节数据
		const uchar* e = src.ptr<uchar>(rows - 1);

		printf("%x,%x,%x\n",e[0], e[1], e[2]);
#endif

#if 0
		//对于单通道图像,每一列存放了1个字节数据
		const uchar* e = src.ptr<uchar>(rows - 1);
		printf("%x,%x\n", e[0], e[1]);
#endif

		//使用api进行直接定位
		if(3 == channels) {
			for (int row = 0; row < rows; row++) {
				for (int cloumn = 0; cloumn < cloumns; cloumn++) {
					//Vec3b:定义一个uchar类型的数组,长度为3而已,<Vect3f>:定义一个float类型的数组,长度为3
					uchar blue = src.at<cv::Vec3b>(row, cloumn)[0];
					uchar green = src.at<cv::Vec3b>(row, cloumn)[1];
					uchar red = src.at<cv::Vec3b>(row, cloumn)[2];

					printf("blue = %x,green = %x,red = %x\n", blue, green, red);
				}
			}
		}
		else if (1 == channels) {
			for (int row = 0; row < rows; row++) {
				for (int cloumn = 0; cloumn < cloumns; cloumn++) {
					//单通道图像,每个像素占用的大小为1个字节
					uchar e = src.at<uchar>(row, cloumn);
					printf("e = %x\n",e);
				}
			}
		}
		else {
		}
		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::waitKey(0);
	}

	/*
		示例4:saturate_cast<uchar>强制转换到的作用

		作用:像素范围处理
		< 0:输出0
		0-255:正常输出
		>255:输出255
	*/
	void improcTest4() {
	
		uchar i = cv::saturate_cast<uchar>(-100);

		uchar j = cv::saturate_cast<uchar>(100);

		uchar k = cv::saturate_cast<uchar>(256);

		printf("i = %d,j = %d,k = %d\n",i,j,k);

		system("pause");

	}

	/*
		示例5:测试图像的深度
		图像的深度:即每个像素占用的字节数

		CV_8U是无符号8位/像素 - 即一个像素的值可以是0-255，这是大多数图像和视频格式的正常范围。

		CV_32F是浮点数 - 像素可以具有0到1.0之间的任何值，这对于数据的某些计算集很有用 - 但是必须通过将每个像素乘以255来将其转换为8位以进行保存或显示。

		CV_32S是每个像素的带符号32位整数值 - 再次对像素进行整数数学运算很有用，但需要转换为8位才能保存或显示。这比较棘手，因为您需要决定如何将更大范围的可能值（+/- 20亿！）转换为0-255

		CV_8UC1，CV_8UC2，CV_8UC3。
	    (最后的1、2、3表示通道数，譬如RGB3通道就用CV_8UC3)
	*/
	void improcTest5() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/02_1.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";

		int channels = src.channels();
		int depth = src.depth();
		int type = src.type();

		printf("channels = %d,depth = %d,type = %d\n", channels, depth,type);

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::waitKey(0);
	}

	/*
		示例6:图像的掩模操作
		作用:实现图像对比度调整,提高对比度
		矩阵的掩膜操作:根据掩膜来重新计算每个像素的像素值,掩膜(mask也被称为Kernel)
		公式:
		I(i,j) = 5 * I(i,j)-[I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1)]
	*/
	void improcTest6() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/03.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string winNameOut = "out image";

		int rows = src.rows; //获取图片行数
		int cloumns = src.cols; //获取图片列数
		int channels = src.channels(); //获取图片通道数

		printf("rows = %d,cloumns = %d,channels = %d\n", rows, cloumns, channels);

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::Mat copyImg;

		/*
			void GpuMat::copyTo(OutputArray dst, InputArray mask)
			作用:图像的拷贝
		*/
		src.copyTo(copyImg);

		if (3 == channels) {
			for (int row = 1; row < (rows-1); row++) {
				for (int cloumn = 1; cloumn < (cloumns-1); cloumn++) {
					//将每一个像素的RGB通道进行赋值
					for (int i = 0; i < channels; i++) {
						copyImg.at<cv::Vec3b>(row, cloumn)[i] = cv::saturate_cast<uchar>(8*src.at<cv::Vec3b>(row, cloumn)[i] - (src.at<cv::Vec3b>(row-1, cloumn)[i]+ src.at<cv::Vec3b>(row+1, cloumn)[i]+ src.at<cv::Vec3b>(row, cloumn-1)[i]+ src.at<cv::Vec3b>(row, cloumn+1)[i]));
					}
				}
			}
		}

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, copyImg);

		cv::waitKey(0);
	}

	/*
		示例7:调用filter2D实现掩膜操作
	*/
	void improcTest7() {
	
		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/03.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string winNameOut = "out image";

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::Mat dst;

		//使用数组生成Mat对象,要使用char类型,不能使用uchar类型,若使用uchar类型-1将变成255,输出结果将错误
		cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

		std::cout << kernel << std::endl;

		/*
		原型:void filter2D( InputArray src, OutputArray dst, int ddepth,
		InputArray kernel, Point anchor = Point(-1,-1),
		double delta = 0, int borderType = BORDER_DEFAULT );

		参数:
		src:源图像
		dst:目标图像
		ddepth:图像的深度
		kernel:掩膜
		*/
		cv::filter2D(src,dst,src.depth(), kernel);

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, dst);

		cv::waitKey(0);
	}

	/*
		示例8:Mat对象
		Mat对象使用:
			部分复制:一般情况下只会复制Mat对象的头和指针部分,不会复制数据部分

			Mat B(A) //浅层复制

			完全复制:
				A.clone()或者A.copyTo(B)
	*/
	void improcTest8() {

		/*
			常用的构造函数
			Mat();
			Mat(int rows, int cols, int type);
			Mat(Size size, int type);
			Mat(int rows, int cols, int type, const Scalar& s);
			Mat(Size size, int type, const Scalar& s);

			cv::Size size(100,100);
		*/
		cv::Mat src;

		/*
			原型:void Mat::create(int _rows, int _cols, int _type)

			CV_8UC3:
				8:表示每个通道占8位
				U:表示无符号数
				C:表示char类型
				3:表示通道数目是3
		*/

		src.create(200,200,CV_8UC3);
		
		//生成颜色值
		src = cv::Scalar(255,255,0);
		
		std::string winName = "in image";
		std::string winNameOut = "out image";

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		printf("weight = %d,heigt = %d\n",src.size().width,src.size().height);

		//定义小数组形式
		//cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

		cv::Mat dst;

		/*
			convertTo():函数负责转换数据类型不同的Mat,即可以将类似float型的Mat转化到imwrite()函数能够接受的类型-数据类型的转换
			cvtColor():函数负责转化不同通道的Mat,一般情况下这个函数用来进行色彩空间的转化-通道的转换
		*/
		src.convertTo(dst, CV_8UC1);

		printf("channels = %d\n", dst.channels());

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, dst);
		cv::waitKey(0);	
	}

	/*
		示例9:图像的混合
		理论:线性混合操作
		g(x) = (1-α)f0(x) + αf1(x)
		其中α的取值范围是0-1之间

		实际应用在中f0(x)可以看做是一幅输入图像,f1(x)可以看做是一幅输入图像,g(x)可以输出图像
	*/
	void improcTest9() {

		cv::Mat src1;
		cv::Mat src2;
		cv::Mat dst;

		std::string path1 = "E:/study/ML/img/pic/improc/09_1.png";
		std::string path2 = "E:/study/ML/img/pic/improc/09_2.png";

		src1 = cv::imread(path1, cv::ImreadModes::IMREAD_COLOR);
		src2 = cv::imread(path2, cv::ImreadModes::IMREAD_COLOR);

		std::cout << "type1 = " << src1.type() << std::endl;
		std::cout << "type2 = " << src2.type() << std::endl;

		double alpha = 0.01;
		double gamma = 0;
		if ((src1.rows == src2.rows) && (src1.cols == src2.cols) && (src1.type() == src2.type())) {
			/*
				void addWeighted(InputArray src1, double alpha, InputArray src2,
                              double beta, double gamma, OutputArray dst, int dtype = -1);

				参数1：输入图像Mat – src1
				参数2：输入图像src1的alpha值
				参数3：输入图像Mat – src2
				参数4：输入图像src2的alpha值
				参数5：gamma值
				参数6：输出混合图像


				注意点：两张图像的大小和类型必须一致才可以

				g(x) = saturate_cast<uchar>((1-α)f0(x) + αf1(x) + gamma)
			*/
			cv::addWeighted(src1,0.5,src2,1- alpha, gamma,dst);
		}

		std::string inName1 = "inName1 image";
		std::string inName2 = "inName2 image";
		std::string winNameOut = "out image";

		cv::namedWindow(inName1, CV_WINDOW_AUTOSIZE);
		cv::imshow(inName1, src1);

		cv::namedWindow(inName2, CV_WINDOW_AUTOSIZE);
		cv::imshow(inName2, src2);

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, dst);

		cv::waitKey(0);
	}

	/*
		示例10:图像对比的调整

		图像变换可以看作如下：
		- 像素变换 – 点操作
		- 邻域操作 – 区域
		调整图像亮度和对比度属于像素变换-点操作

		g(x) = αf0(x) + β,其中α>0,β是增益
	*/
	void improcTest10() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/09_1.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //判断加载数据是否成功
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string outName = "out image";

		//初始化一个同样大小的空白图像
		cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

		int rows = src.rows; //获取图片行数
		int cloumns = src.cols; //获取图片列数
		int channels = src.channels(); //获取图片通道数

		printf("channels = %d\n", channels);

		double k = 1.4;
		double b = 1;

		for (int row = 0; row < rows; row++) {
			for (int cloumn = 0; cloumn < cloumns; cloumn++) {

				uchar blue = src.at<cv::Vec3b>(row, cloumn)[0];
				uchar green = src.at<cv::Vec3b>(row, cloumn)[1];
				uchar red = src.at<cv::Vec3b>(row, cloumn)[2];

				dst.at<cv::Vec3b>(row, cloumn)[0] = cv::saturate_cast<uchar>(k * blue + b);
				dst.at<cv::Vec3b>(row, cloumn)[1] = cv::saturate_cast<uchar>(k * green + b);
				dst.at<cv::Vec3b>(row, cloumn)[2] = cv::saturate_cast<uchar>(k * red + b);
			}
		}

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::namedWindow(outName, CV_WINDOW_AUTOSIZE);
		cv::imshow(outName, dst);

		cv::waitKey(0);
	}
#endif // !_OMPROC_H_

