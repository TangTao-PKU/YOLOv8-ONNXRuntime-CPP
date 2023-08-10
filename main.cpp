#include <iostream>
#include <stdio.h>
#include "inference.h"
#include <filesystem>
#include <opencv2/opencv.hpp>



void file_iterator(DCSP_CORE*& p)
{
	std::filesystem::path img_path = R"(E:/yolov8/examples/samples/test/mix)";
	//std::filesystem::path img_path = R"(E:/yolov8/examples/samples)";
	int k = 0;
	for (auto& i : std::filesystem::directory_iterator(img_path))
	{
		if (i.path().extension() == ".bmp")
		{
			std::string img_path = i.path().string();
			std::cout << img_path << std::endl;
			cv::Mat img = cv::imread(img_path);
			std::vector<DCSP_RESULT> res;
			char* ret = p->RunSession(img, res);
			for (int i = 0; i < res.size(); i++)
			{
				cv::rectangle(img, res.at(i).box, cv::Scalar(125, 123, 0), 3);
			}
			k++;
			cv::imshow("TEST_ORIGIN", img);
			cv::waitKey(0);
			cv::destroyAllWindows();
			cv::imwrite("E:/yolov8/examples/samples/output/" + std::to_string(k) + ".jpg", img);
			
		}
	}
}


int main()
{
	DCSP_CORE* p1 = new DCSP_CORE;
	std::string model_path = "E:/yolov8/examples/samples/best.onnx";
	DCSP_INIT_PARAM params{ model_path, YOLO_CLS_V8, {224, 224}, 2, false };
	//DCSP_INIT_PARAM params{ model_path, YOLO_ORIGIN_V8, {640, 640}, 80, 0.1, 0.5, false };

	/*const int N = 1000;
	for (int i = 0; i < N; i++) {
		char* ret = p1->CreateSession(params);
		file_iterator(p1);
		p1->DestroySession();
	}*/
	char* ret = p1->CreateSession(params);
	file_iterator(p1);


	return 0;
}