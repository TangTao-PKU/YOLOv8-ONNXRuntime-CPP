#include "inference.h"
#include <regex>
#include <iostream>
#include <fstream>
#include <vector>


#define benchmark
#define ELOG

DCSP_CORE::DCSP_CORE()
{

}


DCSP_CORE::~DCSP_CORE()
{
	delete session;
}


template<typename T>
char* BlobFromImage(cv::Mat& iImg, T& iBlob)
{
	int channels = iImg.channels();
	int imgHeight = iImg.rows;
	int imgWidth = iImg.cols;
	//std::cout << channels << std::endl;
	//std::cout << imgHeight << std::endl;
	//std::cout << imgWidth << std::endl;
	for (int c = 0; c < channels; c++)
	{
		for (int h = 0; h < imgHeight; h++)
		{
			for (int w = 0; w < imgWidth; w++)
			{
				iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = (std::remove_pointer<T>::type)((iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
			}
		}
	}

	/*for (int w = 0; w < 10; w++) {
		std::cout << "iImg[0 * imgWidth * imgHeight + 0 * imgWidth + " << w << "] = "
			<< iBlob[0 * imgWidth * imgHeight + 0 * imgWidth + w] << std::endl;
	}*/

	cv::Mat paddedImage(224, 224, CV_8UC3, cv::Scalar(0, 0, 0));

	// Copy the original image into the center of the padded image
	cv::Rect roi((224 - imgWidth) / 2, (224 - imgHeight) / 2, imgWidth, imgHeight);

	iImg.copyTo(paddedImage(roi));

	// Convert to float and normalize
	paddedImage.convertTo(paddedImage, CV_32FC3);

	cv::Scalar mean(0.485, 0.456, 0.406);
	cv::Scalar stdDev(0.229, 0.224, 0.225);
	paddedImage /= 255.0f;
	paddedImage -= mean;
	paddedImage /= stdDev;

	/*for (int w = 0; w < 10; w++) {
		std::cout << "paddedImage[0 * imgWidth * imgHeight + 0 * imgWidth + " << w << "] = "
			<< paddedImage.at<cv::Vec3f>(0, w)[0] << std::endl;
	}

	std::ifstream file("E:/yolov8/examples/tensor_data.bin", std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open file." << std::endl;
	}
	std::vector<float> data(1 * 3 * 224 * 224);
	file.read(reinterpret_cast<char*>(data.data()), 1.0 * 3 * 224 * 224 * sizeof(float));
	file.close();*/

	// 提取数据并打印
	//for (int i = 0; i < 10; i++) {
	//	std::cout << data[i] << " ";
	//	std::cout << std::endl;
	//}


	for (int c = 0; c < channels; c++)
	{
		for (int h = 0; h < imgHeight; h++)
		{
			for (int w = 0; w < imgWidth; w++)
			{
				//iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = (std::remove_pointer<T>::type)((iImg.at<cv::Vec3b>(h, w)[c]) / 255.0f);
				iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = (std::remove_pointer<T>::type)((paddedImage.at<cv::Vec3f>(h, w)[c]));
				//iBlob[c * imgWidth * imgHeight + h * imgWidth + w] = data[c * imgWidth * imgHeight + h * imgWidth + w];
			}
		}
	}

	//for (int w = 0; w < 10; w++) {
	//	std::cout << "iBlob[0 * imgWidth * imgHeight + 0 * imgWidth + " << w << "] = "
	//		<< iBlob[0 * imgWidth * imgHeight + 0 * imgWidth + w] << std::endl;
	//}

	return RET_OK;
}


char* PreProcess(cv::Mat& iImg, std::vector<int> iImgSize, cv::Mat& oImg)
{
	cv::Mat img = iImg.clone();
	cv::resize(iImg, oImg, cv::Size(iImgSize.at(0), iImgSize.at(1)));
	if (img.channels() == 1)
	{
		cv::cvtColor(oImg, oImg, cv::COLOR_GRAY2BGR);
	}
	cv::cvtColor(oImg, oImg, cv::COLOR_BGR2RGB);
	
	return RET_OK;
}


char* DCSP_CORE::CreateSession(DCSP_INIT_PARAM &iParams)
{
	char* Ret = RET_OK;
	std::regex pattern("[\u4e00-\u9fa5]");
	bool result = std::regex_search(iParams.ModelPath, pattern);
	if (result)
	{
		Ret = "[DCSP_ONNX]:model path error.change your model path without chinese characters.";
		std::cout << Ret << std::endl;
		return Ret;
	}
	try
	{
		rectConfidenceThreshold = iParams.RectConfidenceThreshold;
		iouThreshold = iParams.iouThreshold;
		imgSize = iParams.imgSize;
		modelType = iParams.ModelType;
		env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
		Ort::SessionOptions sessionOption;
		if (iParams.CudaEnable)
		{
			cudaEnable = iParams.CudaEnable;
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = 0;
			std::cout << "success" << std::endl;
			sessionOption.AppendExecutionProvider_CUDA(cudaOption);
			//OrtOpenVINOProviderOptions ovOption;
			//sessionOption.AppendExecutionProvider_OpenVINO(ovOption);
			std::cout << "success" << std::endl;
		}
		sessionOption.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		sessionOption.SetIntraOpNumThreads(iParams.IntraOpNumThreads);
		sessionOption.SetLogSeverityLevel(iParams.LogSeverityLevel);
		int ModelPathSize = MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), nullptr, 0);
		wchar_t* wide_cstr = new wchar_t[ModelPathSize + 1];
		MultiByteToWideChar(CP_UTF8, 0, iParams.ModelPath.c_str(), static_cast<int>(iParams.ModelPath.length()), wide_cstr, ModelPathSize);
		wide_cstr[ModelPathSize] = L'\0';
		const wchar_t* modelPath = wide_cstr;
		session = new Ort::Session(env, modelPath, sessionOption);

		

		Ort::AllocatorWithDefaultOptions allocator;
		size_t inputNodesNum = session->GetInputCount();
		for (size_t i = 0; i < inputNodesNum; i++)
		{
			Ort::AllocatedStringPtr input_node_name = session->GetInputNameAllocated(i, allocator);
			char* temp_buf = new char[50];
			strcpy(temp_buf, input_node_name.get());
			inputNodeNames.push_back(temp_buf);
		}

		size_t OutputNodesNum = session->GetOutputCount();
		for (size_t i = 0; i < OutputNodesNum; i++)
		{
			Ort::AllocatedStringPtr output_node_name = session->GetOutputNameAllocated(i, allocator);
			char* temp_buf = new char[10];
			strcpy(temp_buf, output_node_name.get());
			outputNodeNames.push_back(temp_buf);
		}
		options = Ort::RunOptions{ nullptr };
		WarmUpSession();
		//std::cout << OrtGetApiBase()->GetVersionString() << std::endl;	//1.15.1
		Ret = RET_OK;
		return Ret;
	}
	catch (const std::exception& e)
	{
		const char* str1 = "[DCSP_ONNX]:";
		const char* str2 = e.what();
		std::string result = std::string(str1) + std::string(str2);
		char* merged = new char[result.length() + 1];
		std::strcpy(merged, result.c_str());
		std::cout << merged << std::endl;
		delete[] merged;
		//return merged;
		return "[DCSP_ONNX]:Create session failed.";
	}

}


char* DCSP_CORE::RunSession(cv::Mat &iImg, std::vector<DCSP_RESULT>& oResult)
{
#ifdef benchmark
	clock_t starttime_1 = clock();
#endif // benchmar
	char* Ret = RET_OK;
	cv::Mat processedImg;
	PreProcess(iImg, imgSize, processedImg);
	if (modelType < 4)
	{
		float* blob = new float[processedImg.total() * 3];
		//BlobFromImage(processedImg, blob);
		BlobFromImage(processedImg, blob);
		std::vector<int64_t> inputNodeDims = { 1,3,imgSize.at(0),imgSize.at(1) };
		TensorProcess(starttime_1, iImg, blob, inputNodeDims, oResult);
	}
	return Ret;
}


template<typename N>
char* DCSP_CORE::TensorProcess(clock_t& starttime_1, cv::Mat& iImg, N& blob, std::vector<int64_t>& inputNodeDims, std::vector<DCSP_RESULT>& oResult)
{
	Ort::Value inputTensor = Ort::Value::CreateTensor<std::remove_pointer<N>::type>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), inputNodeDims.data(), inputNodeDims.size());
#ifdef benchmark
	clock_t starttime_2 = clock();
#endif // benchmark
	auto outputTensor = session->Run(options, inputNodeNames.data(), &inputTensor, 1, outputNodeNames.data(), outputNodeNames.size());
#ifdef benchmark
	clock_t starttime_3 = clock();
#endif // benchmark
	Ort::TypeInfo typeInfo = outputTensor.front().GetTypeInfo();
	auto tensor_info = typeInfo.GetTensorTypeAndShapeInfo();
	std::vector<int64_t>outputNodeDims = tensor_info.GetShape();
	std::remove_pointer<N>::type* output = outputTensor.front().GetTensorMutableData<std::remove_pointer<N>::type>();


	//输出维度
	//std::cout << outputNodeDims[0] << std::endl;
	//std::cout << outputNodeDims[1] << std::endl;
	/*for (int i = 1; i <= 1000; i++) {
		if (output[i] > 0.1) {
			std::cout << i << std::endl;
		}
	}*/
	//float max = 0;
	//int maxidx = 0;
	//float sum = 0;
	//for (int i = 0; i < 1000; i++) {
	//	sum += output[i];
	//	if (output[i] > max) {
	//		max = output[i];
	//		maxidx = i;
	//	}
	//	/*std::cout << output[i] << " ";
	//	if (i%10 == 0) {
	//		std::cout << std::endl;
	//	}*/
	//}
	//分类最大值
	//std::cout << sum << std::endl;
	//std::cout << max << std::endl;
	//std::cout << maxidx << std::endl;

	delete blob;
	switch (modelType)
	{
	case 1:
	{
		int strideNum = outputNodeDims[2];
		int signalResultNum = outputNodeDims[1];
		std::vector<int> class_ids; 
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;
		cv::Mat rowData(signalResultNum, strideNum, CV_32F, output);
		rowData = rowData.t();

		float* data = (float*)rowData.data;

		float x_factor = iImg.cols / 640.;
		float y_factor = iImg.rows / 640.;
		for (int i = 0; i < strideNum; ++i)
		{
			float* classesScores = data + 4;
			cv::Mat scores(1, classesNum, CV_32FC1, classesScores);
			cv::Point class_id;
			double maxClassScore;
			cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
			if (maxClassScore > rectConfidenceThreshold)
			{
				confidences.push_back(maxClassScore);
				class_ids.push_back(class_id.x);

				float x = data[0];
				float y = data[1];
				float w = data[2];
				float h = data[3];

				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);

				int width = int(w * x_factor);
				int height = int(h * y_factor);

				boxes.push_back(cv::Rect(left, top, width, height));
			}
			data += signalResultNum;
		}

		std::vector<int> nmsResult;
		cv::dnn::NMSBoxes(boxes, confidences, rectConfidenceThreshold, iouThreshold, nmsResult);
		for (int i = 0; i < nmsResult.size(); ++i)
		{
			int idx = nmsResult[i];
			DCSP_RESULT result;
			result.classId = class_ids[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx];
			oResult.push_back(result);
		}


#ifdef benchmark
		clock_t starttime_4 = clock();
		double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
		double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
		double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
		if (cudaEnable)
		{
			std::cout << "[DCSP_ONNX(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
		}
		else
		{
			std::cout << "[DCSP_ONNX(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
		}
#endif // benchmark

		break;
	}
	case 3:
	{
		//后处理
		std::cout << "ng: " << output[0] << std::endl;
		std::cout << "ok: " << output[1] << std::endl;
#ifdef benchmark
		clock_t starttime_4 = clock();
		double pre_process_time = (double)(starttime_2 - starttime_1) / CLOCKS_PER_SEC * 1000;
		double process_time = (double)(starttime_3 - starttime_2) / CLOCKS_PER_SEC * 1000;
		double post_process_time = (double)(starttime_4 - starttime_3) / CLOCKS_PER_SEC * 1000;
		if (cudaEnable)
		{
			std::cout << "[DCSP_ONNX(CUDA)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
		}
		else
		{
			std::cout << "[DCSP_ONNX(CPU)]: " << pre_process_time << "ms pre-process, " << process_time << "ms inference, " << post_process_time << "ms post-process." << std::endl;
		}
#endif // benchmark
		break;
	}
	}
	char* Ret = RET_OK;
	return Ret;
}


char* DCSP_CORE::WarmUpSession()
{
	clock_t starttime_1 = clock();
	char* Ret = RET_OK;
	cv::Mat iImg = cv::Mat(cv::Size(imgSize.at(0), imgSize.at(1)), CV_8UC3);
	cv::Mat processedImg;
	PreProcess(iImg, imgSize, processedImg);
	if (modelType < 4)
	{
		float* blob = new float[iImg.total() * 3];
		BlobFromImage(processedImg, blob);
		std::vector<int64_t> YOLO_input_node_dims = { 1,3,imgSize.at(0),imgSize.at(1) };
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU), blob, 3 * imgSize.at(0) * imgSize.at(1), YOLO_input_node_dims.data(), YOLO_input_node_dims.size());
		auto output_tensors = session->Run(options, inputNodeNames.data(), &input_tensor, 1, outputNodeNames.data(), outputNodeNames.size());
		delete[] blob;
		clock_t starttime_4 = clock();
		double post_process_time = (double)(starttime_4 - starttime_1) / CLOCKS_PER_SEC * 1000;
		if (cudaEnable)
		{
			std::cout << "[DCSP_ONNX(CUDA)]: " << "Cuda warm-up cost " << post_process_time << " ms. " << std::endl;
		}
	}

	return Ret;
}

void DCSP_CORE::DestroySession()
{
	if (session) {
		delete session;
	}
}