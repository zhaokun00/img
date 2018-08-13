#ifndef _OMPROC_H_

	#define _OMPROC_H_

	#include <stdio.h>
	#include<stdlib.h>
	#include <iostream>
	#include <math.h>
	#include <opencv2/opencv.hpp>

	/*
		ʾ��1:����ͼƬ�Ķ�ȡ����ʾ
	*/
	void improcTest1() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/01.png";

		/*
			ԭ��:imread( const String& filename, int flags = IMREAD_COLOR );

			����:
				����ͼ��
			����:
				filename:ͼ���ļ�����
				flags:���ص�ͼ��ʱʲô����,֧�ֳ�����3������ֵ

				IMREAD_UNCHANGED            = -1, //��ʾ����ԭͼ,�����κθı�
				IMREAD_GRAYSCALE            = 0,  //��ʾ��ԭͼ��Ϊ�Ҷ�ͼ����ؽ���
				IMREAD_COLOR                = 1,  //��ʾ��ԭͼ��ΪRGBͼ����ؽ���

		*/

		//src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);
		//src = cv::imread(path, cv::ImreadModes::IMREAD_GRAYSCALE);
		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

#if 0
		if (!src.data) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}
#endif
		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		/*
			ԭ��: void namedWindow(const String& winname, int flags = WINDOW_AUTOSIZE);
			
			����:
				����һ��OpenCV����,������OpenCv�Զ��������ͷ�,������Ϊ������
			����:
				winname:��������
				flags:��������
				CV_WINDOW_AUTOSIZE:�Զ�����ͼ���С,��ʾ���ڴ�С,������Ϊ�ı䴰�ڴ�С
				CV_WINDOW_NORMAL:�����޸Ĵ��ڴ�С
		*/

		std::string winName = "input image";
		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);

		/*
			ԭ��:void imshow(const String& winname, InputArray mat)

			����:��ʾͼ��
			����:
				winname:��������
				mat:Mat����
		*/
		cv::imshow("input image", src);

		/*
			ԭ��:int waitKey(int delay = 0);
			
			waitKey()�����������һ��������ʱ����(��λms)�ȴ��û���������,��������õ�ʱ����û�а�������,�����
			waitKey(0)��ʾ����������Ƶĵȴ��û��İ����¼�,���������������,����һֱ�ȴ�
		*/
		cv::waitKey(0);
	}

	/*
		ʾ��2:����ɫ�ʿռ��ת���ͱ���ͼƬ
	*/
	void improcTest2() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/01.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		std::string winName = "out image";

		cv::Mat dst;

		/*
			ԭ��:cvtColor( InputArray src, OutputArray dst, int code, int dstCn = 0 )

			����:��ͼ���һ��ɫ�ʿռ�ת��������һ��ɫ�ʿռ�
			����:
				src:Դͼ��
				dst:Ŀ��ͼ��
				code:ת����ɫ�ʿռ�
		*/
		cv::cvtColor(src,dst,CV_BGR2GRAY);
		//cv::cvtColor(src, dest, CV_BGR2HLS);

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, dst);

		std::string outPath = "E:/study/ML/img/pic/improc/01_out.png";
		/*
			ԭ��:bool imwrite( const String& filename, InputArray img,const std::vector<int>& params = std::vector<int>());
			����:
				����ͼ��
			����:
				filename:�ļ���
				img:Mat����
			ֻ��8λ��16λ��PNG��jpg��tiff�ļ���ʽ�����ǵ�ͨ��������ͨ����BRG��ͼ��ſ���ͨ�����ַ�ʽ����
		*/
		cv::imwrite(outPath,dst);

		cv::waitKey(0);
	}

	/*
		ʾ��3:���Զ�ȡͼƬ��Ԫ��

		��һ��GRAY���ص������ֵ��CV_8UC1��
		Scalar intensity = img.at<uchar>(y, x);
		���� Scalar intensity = img.at<uchar>(Point(x, y));

		��һ��RGB���ص������ֵ
		Vec3f intensity = img.at<Vec3f>(y, x);
		float blue = intensity.val[0];
		float green = intensity.val[1];
		float red = intensity.val[2];

		Vec3b��Ӧ��ͨ����˳����blue��green��red��uchar�������ݡ�
		Vec3f��Ӧ��ͨ����float��������
		��CV_8UC1ת����CV32F1ʵ�����£�
		src.convertTo(dst, CV_32F);
	*/
	void improcTest3() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/02_2.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";

		int rows = src.rows; //��ȡͼƬ����
		int cloumns = src.cols; //��ȡͼƬ����
		int channels = src.channels(); //��ȡͼƬͨ����

		printf("rows = %d,cloumns = %d,channels = %d\n",rows,cloumns,channels);

#if 0
		//����RGB��ͼ��,ÿһ�д����3�ֽ�����
		const uchar* e = src.ptr<uchar>(rows - 1);

		printf("%x,%x,%x\n",e[0], e[1], e[2]);
#endif

#if 0
		//���ڵ�ͨ��ͼ��,ÿһ�д����1���ֽ�����
		const uchar* e = src.ptr<uchar>(rows - 1);
		printf("%x,%x\n", e[0], e[1]);
#endif

		//ʹ��api����ֱ�Ӷ�λ
		if(3 == channels) {
			for (int row = 0; row < rows; row++) {
				for (int cloumn = 0; cloumn < cloumns; cloumn++) {
					//Vec3b:����һ��uchar���͵�����,����Ϊ3����,<Vect3f>:����һ��float���͵�����,����Ϊ3
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
					//��ͨ��ͼ��,ÿ������ռ�õĴ�СΪ1���ֽ�
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
		ʾ��4:saturate_cast<uchar>ǿ��ת����������

		����:���ط�Χ����
		< 0:���0
		0-255:�������
		>255:���255
	*/
	void improcTest4() {
	
		uchar i = cv::saturate_cast<uchar>(-100);

		uchar j = cv::saturate_cast<uchar>(100);

		uchar k = cv::saturate_cast<uchar>(256);

		printf("i = %d,j = %d,k = %d\n",i,j,k);

		system("pause");

	}

	/*
		ʾ��5:����ͼ������
		ͼ������:��ÿ������ռ�õ��ֽ���

		CV_8U���޷���8λ/���� - ��һ�����ص�ֵ������0-255�����Ǵ����ͼ�����Ƶ��ʽ��������Χ��

		CV_32F�Ǹ����� - ���ؿ��Ծ���0��1.0֮����κ�ֵ����������ݵ�ĳЩ���㼯������ - ���Ǳ���ͨ����ÿ�����س���255������ת��Ϊ8λ�Խ��б������ʾ��

		CV_32S��ÿ�����صĴ�����32λ����ֵ - �ٴζ����ؽ���������ѧ��������ã�����Ҫת��Ϊ8λ���ܱ������ʾ����Ƚϼ��֣���Ϊ����Ҫ������ν�����Χ�Ŀ���ֵ��+/- 20�ڣ���ת��Ϊ0-255

		CV_8UC1��CV_8UC2��CV_8UC3��
	    (����1��2��3��ʾͨ������Ʃ��RGB3ͨ������CV_8UC3)
	*/
	void improcTest5() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/02_1.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
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
		ʾ��6:ͼ�����ģ����
		����:ʵ��ͼ��Աȶȵ���,��߶Աȶ�
		�������Ĥ����:������Ĥ�����¼���ÿ�����ص�����ֵ,��Ĥ(maskҲ����ΪKernel)
		��ʽ:
		I(i,j) = 5 * I(i,j)-[I(i-1,j)+I(i+1,j)+I(i,j-1)+I(i,j+1)]
	*/
	void improcTest6() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/03.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string winNameOut = "out image";

		int rows = src.rows; //��ȡͼƬ����
		int cloumns = src.cols; //��ȡͼƬ����
		int channels = src.channels(); //��ȡͼƬͨ����

		printf("rows = %d,cloumns = %d,channels = %d\n", rows, cloumns, channels);

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::Mat copyImg;

		/*
			void GpuMat::copyTo(OutputArray dst, InputArray mask)
			����:ͼ��Ŀ���
		*/
		src.copyTo(copyImg);

		if (3 == channels) {
			for (int row = 1; row < (rows-1); row++) {
				for (int cloumn = 1; cloumn < (cloumns-1); cloumn++) {
					//��ÿһ�����ص�RGBͨ�����и�ֵ
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
		ʾ��7:����filter2Dʵ����Ĥ����
	*/
	void improcTest7() {
	
		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/03.png";

		src = cv::imread(path, cv::ImreadModes::IMREAD_COLOR);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string winNameOut = "out image";

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		cv::Mat dst;

		//ʹ����������Mat����,Ҫʹ��char����,����ʹ��uchar����,��ʹ��uchar����-1�����255,������������
		cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

		std::cout << kernel << std::endl;

		/*
		ԭ��:void filter2D( InputArray src, OutputArray dst, int ddepth,
		InputArray kernel, Point anchor = Point(-1,-1),
		double delta = 0, int borderType = BORDER_DEFAULT );

		����:
		src:Դͼ��
		dst:Ŀ��ͼ��
		ddepth:ͼ������
		kernel:��Ĥ
		*/
		cv::filter2D(src,dst,src.depth(), kernel);

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, dst);

		cv::waitKey(0);
	}

	/*
		ʾ��8:Mat����
		Mat����ʹ��:
			���ָ���:һ�������ֻ�Ḵ��Mat�����ͷ��ָ�벿��,���Ḵ�����ݲ���

			Mat B(A) //ǳ�㸴��

			��ȫ����:
				A.clone()����A.copyTo(B)
	*/
	void improcTest8() {

		/*
			���õĹ��캯��
			Mat();
			Mat(int rows, int cols, int type);
			Mat(Size size, int type);
			Mat(int rows, int cols, int type, const Scalar& s);
			Mat(Size size, int type, const Scalar& s);

			cv::Size size(100,100);
		*/
		cv::Mat src;

		/*
			ԭ��:void Mat::create(int _rows, int _cols, int _type)

			CV_8UC3:
				8:��ʾÿ��ͨ��ռ8λ
				U:��ʾ�޷�����
				C:��ʾchar����
				3:��ʾͨ����Ŀ��3
		*/

		src.create(200,200,CV_8UC3);
		
		//������ɫֵ
		src = cv::Scalar(255,255,0);
		
		std::string winName = "in image";
		std::string winNameOut = "out image";

		cv::namedWindow(winName, CV_WINDOW_AUTOSIZE);
		cv::imshow(winName, src);

		printf("weight = %d,heigt = %d\n",src.size().width,src.size().height);

		//����С������ʽ
		//cv::Mat kernel = (cv::Mat_<char>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

		cv::Mat dst;

		/*
			convertTo():��������ת���������Ͳ�ͬ��Mat,�����Խ�����float�͵�Matת����imwrite()�����ܹ����ܵ�����-�������͵�ת��
			cvtColor():��������ת����ͬͨ����Mat,һ����������������������ɫ�ʿռ��ת��-ͨ����ת��
		*/
		src.convertTo(dst, CV_8UC1);

		printf("channels = %d\n", dst.channels());

		cv::namedWindow(winNameOut, CV_WINDOW_AUTOSIZE);
		cv::imshow(winNameOut, dst);
		cv::waitKey(0);	
	}

	/*
		ʾ��9:ͼ��Ļ��
		����:���Ի�ϲ���
		g(x) = (1-��)f0(x) + ��f1(x)
		���Ц���ȡֵ��Χ��0-1֮��

		ʵ��Ӧ������f0(x)���Կ�����һ������ͼ��,f1(x)���Կ�����һ������ͼ��,g(x)�������ͼ��
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

				����1������ͼ��Mat �C src1
				����2������ͼ��src1��alphaֵ
				����3������ͼ��Mat �C src2
				����4������ͼ��src2��alphaֵ
				����5��gammaֵ
				����6��������ͼ��


				ע��㣺����ͼ��Ĵ�С�����ͱ���һ�²ſ���

				g(x) = saturate_cast<uchar>((1-��)f0(x) + ��f1(x) + gamma)
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
		ʾ��10:ͼ��Աȵĵ���

		ͼ��任���Կ������£�
		- ���ر任 �C �����
		- ������� �C ����
		����ͼ�����ȺͶԱȶ��������ر任-�����

		g(x) = ��f0(x) + ��,���Ц�>0,��������
	*/
	void improcTest10() {

		cv::Mat src;
		std::string path = "E:/study/ML/img/pic/improc/09_1.bmp";

		src = cv::imread(path, cv::ImreadModes::IMREAD_UNCHANGED);

		if (true == src.empty()) { //�жϼ��������Ƿ�ɹ�
			printf("could not load image...\n");
			return;
		}

		std::string winName = "in image";
		std::string outName = "out image";

		//��ʼ��һ��ͬ����С�Ŀհ�ͼ��
		cv::Mat dst = cv::Mat::zeros(src.size(),src.type());

		int rows = src.rows; //��ȡͼƬ����
		int cloumns = src.cols; //��ȡͼƬ����
		int channels = src.channels(); //��ȡͼƬͨ����

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

