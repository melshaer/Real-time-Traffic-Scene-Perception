//Generating Test Samples from a trained RBM
//Menna El-Shaer -- March 2018

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <random>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//CUDA
#include "helper_cuda.h"
#include <cuda_runtime.h>

//Constants for fast random number generation (Taken from "Deep Belief Nets in CUDA C/C++" by Timothy Masters)
#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836

static cudaDeviceProp deviceProps;

//Define host variables
//static float *h_w;
//static float *h_wtr;
//static float *h_vis_bias;
//static float *h_hid_bias;


//Define global variables in unified memory space
__device__ __managed__ int n_vis;					//Number of visible neurons
__device__ __managed__ int n_hid;					//Number of hidden neurons
__device__ __managed__ int n_chain;					//Length of Gibbs Chain

//Function Declarations
int rbm_sample_test(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer, const int num_rows_crop, const int num_cols_crop);
int vis_to_hid(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer);
int hid_to_vis(float *w, float *wtr, float *vis_bias, float *hid_bias, float *hid_layer, float *vis_layer);

//Kernel Functions Declarations
__global__ void vis_to_hid_kernel(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer);
__global__ void hid_to_vis_kernel(float *w, float *wtr, float *vis_bias, float *hid_bias, float *hid_layer, float *vis_layer);


int rbm_sample_test(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer, const int num_rows_crop, const int num_cols_crop)
{
	//Initialize CUDA, including sending all data to device
	std::cout << "Initializing CUDA for testing..." << std::endl;

	int device = 0;
	int *cuda_device = &device;

	checkCudaErrors(cudaSetDevice(*cuda_device));
	checkCudaErrors(cudaGetDeviceProperties(&deviceProps, *cuda_device));

	std::cout << "CUDA initialization for testing ended successfully" << std::endl;

	const int n_step = 1;

	cudaEvent_t startTimer;
	cudaEvent_t stopTimer;

	cudaEventCreate(&startTimer);
	cudaEventCreate(&stopTimer);

	//Gibbs Chain (iterative)
	for(int istep = 0; istep < n_step; istep++)
	{
		cudaEventRecord(startTimer);
		//At each new Gibbs step, initialize new visible layer image with the last
		std::cout << "Starting Gibbs step " << istep <<std::endl;

		//Sample visible to hidden -- with sampling; for each hidden unit, find its visible layer activation
		int ret_val = vis_to_hid(w, wtr, vis_bias, hid_bias, vis_layer, hid_layer);
		if(ret_val)
			std::cout << "Visible to Hidden sampling function failed at Gibbs step " << istep << std::endl;

		//Sample hidden to visible -- without sampling
		ret_val = hid_to_vis(w, wtr, vis_bias, hid_bias, hid_layer, vis_layer);
		if(ret_val)
			std::cout << "Hidden to Visible sampling function failed at Gibbs step " << istep << std::endl;

			/*for(int j = 0; j < n_vis; j++){
				float diff = vis_layer[n_vis + j] - vis_layer[j];
			std::cout << "Difference is " << diff << std::endl;
			}*/

		cudaEventRecord(stopTimer);
		cudaEventSynchronize(stopTimer);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, startTimer, stopTimer);
		std::cout << "Gibbs step " << istep << " took " << milliseconds << "ms of time " << std::endl;

		//Display and save generated sample every 1000 Gibbs iterations
		if(istep == 0)
		{
			cv::Mat chain_samples(2*n_chain, n_vis, CV_32F, vis_layer);
			cv::Mat sample(num_rows_crop, num_cols_crop, CV_32F);

			for (int i = 0; i < 2*n_chain; i++)
			{
				double maxVal, minVal;
				const float *row_ptr = chain_samples.ptr<float>(i);

				for(int j = 0; j < num_rows_crop; j++)
					for(int k = 0; k < num_cols_crop; k++)
						sample.at<float>(j,k) = row_ptr[k+j*num_cols_crop];

				cv::minMaxLoc(sample, &minVal, &maxVal);
				std::cout << "Min = " << minVal << " " << "Max = " << maxVal << std::endl;
				sample.convertTo(sample, CV_32F, (1/(maxVal-minVal)), (-minVal/(maxVal-minVal)));
				//cv::imshow("Generated Sample for Chain " + std::to_string(i), sample);
				//cv::waitKey(15);
				sample.convertTo(sample, CV_32F, 255.0);
				if(i == 0){
				cv::imwrite(("/home/ubuntu/ubuntu/RBM_Generative/Generated_Samples/PG/Samples_for_quality/Hidden_"
						+ std::to_string(n_hid) + "Nodes" + ".png").c_str(), sample);}
			}
		}
	} //Next Gibbs Step

	return 0;
}


__global__ void vis_to_hid_kernel(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer)
{
	int ihid = blockIdx.x * blockDim.x + threadIdx.x;
	if (ihid >= n_hid)
		return;

	int ichain = blockIdx.y;

	float sum = hid_bias[ihid];
	int randnum = 1;

	for (int ivis = 0; ivis < n_vis; ivis++)
	{
		if (vis_layer[ichain*n_vis+ivis] > 0.5)
			++randnum;
		sum += wtr[ivis*n_hid+ihid] * vis_layer[ichain*n_vis+ivis];
	}

	sum = sum / n_vis;
	//printf("Sum = %f", sum);

	float act_Q = 1.0f / (1.0f + __expf(-sum));

	int k = randnum / IQ;
	randnum = IA * (randnum - k * IQ) - IR * k;
	if (randnum < 0)
		randnum += IM;
	float frand = AM * randnum;
	hid_layer[ichain*n_hid+ihid] = (frand < act_Q) ? 1.0 : 0.0;

}

int vis_to_hid(float *w, float *wtr, float *vis_bias, float *hid_bias, float *vis_layer, float *hid_layer)
{
	dim3 block_launch;
	cudaError_t error_id;

	cudaEvent_t startKernelTimer;
	cudaEvent_t stopKernelTimer;
	float milliseconds = 0;

	cudaEventCreate(&startKernelTimer);
	cudaEventCreate(&stopKernelTimer);

	//int warpsize = deviceProps.warpSize;      //Threads per warp

	/*for (int i = 0; i<n_chain; i++)
				for(int j = 0; j < n_vis; j++)
					std::cout << "vis_layer at index " << i*n_vis + j << " is " << vis_layer[i*n_vis + j];*/

//	int threads_per_block = (n_hid + warpsize - 1) / warpsize * warpsize;
//	if (threads_per_block > 4 * warpsize)
//		threads_per_block = 4 * warpsize;
	int threads_per_block = 1024;

	block_launch.x = (n_hid + threads_per_block - 1) / threads_per_block;
	block_launch.y = 2*n_chain;
	block_launch.z = 1;

	printf("vis_to_hid kernel launched with block size = %d and grid sizex = %d \n", threads_per_block, block_launch.x);
	cudaEventRecord(startKernelTimer);
	vis_to_hid_kernel<<< block_launch, threads_per_block >>>(w, wtr, vis_bias, hid_bias, vis_layer, hid_layer);
	cudaEventRecord(stopKernelTimer);

	cudaDeviceSynchronize();
	cudaEventSynchronize(stopKernelTimer);

	cudaEventElapsedTime(&milliseconds, startKernelTimer, stopKernelTimer);
	std::cout << "vis to hid kernel took " << milliseconds << "ms of time " << std::endl;


//	for (int i = 0; i<2*n_chain; i++)
//		for(int j = 0; j < n_hid; j++)
//			std::cout << "hid_layer at index " << i*n_hid + j << " is " << hid_layer[i*n_hid + j] << std::endl;

	error_id = cudaGetLastError();
	   if (error_id != cudaSuccess)
		   return 1;
	   else
		   return 0;

}


__global__ void hid_to_vis_kernel(float *w, float *wtr, float *vis_bias, float *hid_bias, float *hid_layer, float *vis_layer)
{
	int ivis = blockIdx.x * blockDim.x + threadIdx.x;
	if (ivis >= n_vis)
		return;

	int ichain = blockIdx.y;

	float sum = vis_bias[ivis];

	for (int ihid = 0; ihid < n_hid; ihid++){
		sum += w[ihid*n_vis+ivis] * hid_layer[ichain*n_hid+ihid];
		//printf("Weight = %f\n", w[ihid*n_vis+ivis]);
	}
	//printf("Sum2 = %f", sum);

	vis_layer[ichain*n_vis+ivis] = 1.0 / (1.0 + __expf(-sum));

}


int hid_to_vis(float *w, float *wtr, float *vis_bias, float *hid_bias, float *hid_layer, float *vis_layer)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (n_vis + warpsize - 1) / warpsize * warpsize;
		if (threads_per_block > 4 * warpsize)
			threads_per_block = 4 * warpsize;

	block_launch.x = (n_vis + threads_per_block - 1) / threads_per_block;
	block_launch.y = 2*n_chain;
	block_launch.z = 1;

	printf("hid_to_vis kernel launched with block size = %d and grid sizex = %d \n", threads_per_block, block_launch.x);
	hid_to_vis_kernel <<< block_launch, threads_per_block >>>(w, wtr, vis_bias, hid_bias, hid_layer, vis_layer);
	cudaDeviceSynchronize();

	/*for (int i = 0; i<n_chain; i++)
		for(int j = 0; j < n_vis; j++)
			std::cout << "vis_layer at index " << i*n_vis + j << " is " << vis_layer[i*n_vis + j] << std::endl;*/

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess)
		return 1;
	else
		return 0;
}


int main(int argc, char** argv)
{
	cudaDeviceReset(); //Flush Profiler Data
	//Prepare test data
	//const int num_frames_0 = 418;
	//const int num_frames_1 = 558;
	//const int num_frames = num_frames_0 + num_frames_1;
	const int num_frames = 4080;  //maximum frame index in dataset

	const int num_rows_crop = 360;
	const int num_cols_crop = 960;
	const int num_pixels = num_rows_crop * num_cols_crop;
	const int x_crop = 160;
	const int y_crop = 180;
	cv::Rect toCrop = cv::Rect(x_crop, y_crop, num_cols_crop, num_rows_crop);

	n_chain = 20;     //Number of Gibbs chains, i.e. number of test images used
	n_hid = 45;
	n_vis = num_pixels;

	int index = 0;

	float *vis_layer;
	cudaMallocManaged(&vis_layer, 2*n_chain*n_vis*sizeof(float));

	float *hid_layer;
	cudaMallocManaged(&hid_layer, 2*n_chain*n_hid*sizeof(float));

	//Pick n_chains random images to use as test, get Mat, transform and add to data array
	cv::String fileName_0, fileName_1;
	cv::String folderPath_0 = "/home/ubuntu/ubuntu/RBM_Generative/Test_Combined/Camera_0/Frame_";
	cv::String folderPath_1 = "/home/ubuntu/ubuntu/RBM_Generative/Test_Combined/Camera_1/Frame_";

	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<> distribution(1, num_frames);

	while(index < 2*n_chain)
	{
		double minVal, maxVal;
		int frame_0 = distribution(generator);
		int frame_1 = distribution(generator);
		std::cout << "Frame 0: " << frame_0 << std::endl;
		std::cout << "Frame 1: " << frame_1 << std::endl;

		//Read and Crop
		fileName_0 = folderPath_0 + std::to_string(frame_0) + ".jpg";
		fileName_1 = folderPath_1 + std::to_string(frame_1) + ".jpg";
		cv::Mat bgr_image_0 = cv::imread(fileName_0.c_str(), CV_LOAD_IMAGE_COLOR);
		cv::Mat bgr_image_1 = cv::imread(fileName_1.c_str(), CV_LOAD_IMAGE_COLOR);

		if(bgr_image_0.data != NULL)
		{
			cv::Mat cropped_ref = bgr_image_0(toCrop); //creates a reference only
			cv::Mat cropped_temp;
			cropped_ref.copyTo(cropped_temp);

			//Grayscale the cropped image
			cv::Mat cropped;
			cv::cvtColor(cropped_temp, cropped, CV_BGR2GRAY);
			//cv::imshow("GS", cropped);
			//cv::waitKey(1);

			//Normalize (Scale) and convert to float(4 bytes)
			cropped.convertTo(cropped, CV_32F, 1.0/255.0);

			cv::minMaxLoc(cropped, &minVal, &maxVal);
			std::cout << "Min = " << minVal << " " << "Max = " << maxVal << std::endl;

			//Copy cropped 2D test image into the test image array
			for(int j = 0; j < cropped.rows; j++)
			{
				int i = j * cropped.cols;
				const float* row_ptr = cropped.ptr<float>(j);
				for(int k = 0; k < cropped.cols; k++)
					vis_layer[(index*n_vis+i)+k] = row_ptr[k];
			}
			index++;
			std::cout << "Index++ Image from Camera_0 added to vis_layer" << std::endl;
		}

		if(bgr_image_1.data != NULL)
		{
			cv::Mat cropped_ref = bgr_image_1(toCrop); //creates a reference only
			cv::Mat cropped_temp;
			cropped_ref.copyTo(cropped_temp);

			//Grayscale the cropped image
			cv::Mat cropped;
			cv::cvtColor(cropped_temp, cropped, CV_BGR2GRAY);
			//cv::imshow("GS", cropped);
			//cv::waitKey(1);

			//Normalize (Scale) and convert to float(4 bytes)
			cropped.convertTo(cropped, CV_32F, 1.0/255.0);

			cv::minMaxLoc(cropped, &minVal, &maxVal);
			std::cout << "Min = " << minVal << " " << "Max = " << maxVal << std::endl;

			//Copy cropped 2D test image into the test image array
			for(int j = 0; j < cropped.rows; j++)
			{
				int i = j * cropped.cols;
				const float* row_ptr = cropped.ptr<float>(j);
				for(int k = 0; k < cropped.cols; k++)
					vis_layer[(index*n_vis+i)+k] = row_ptr[k];
			}
			index++;
			std::cout << "Index++ Image from Camera_1 added to vis_layer" << std::endl;
		}
	}

	std::cout << "Final index is " << index << std::endl;

	//Zero mean, unit variance?

	/*for(int j = 0; j < n_vis; j++){
					float diff = vis_layer[n_vis + j] - vis_layer[j];
				std::cout << "Difference is " << diff << std::endl;
				}*/

	//Read in trained parameters
	std::string w_fileName = "/home/ubuntu/ubuntu/RBM_Generative/Trained_Parameters/Training_Combined_4/weights_" +
    		std::to_string(n_hid) + "hidden_units.txt";
	std::string vis_fileName = "/home/ubuntu/ubuntu/RBM_Generative/Trained_Parameters/Training_Combined_4/vis_bias_" +
    		std::to_string(n_hid) + "hidden_units.txt";
	std::string hid_fileName = "/home/ubuntu/ubuntu/RBM_Generative/Trained_Parameters/Training_Combined_4/hid_bias_" +
    		std::to_string(n_hid) + "hidden_units.txt";

	//Define file streams
	std::ifstream w_if(w_fileName);
	std::ifstream vis_if(vis_fileName);
	std::ifstream hid_if(hid_fileName);

	//h_w = new float[n_hid*n_vis];
	//h_wtr = new float[n_vis*n_hid];
	//h_vis_bias = new float[n_vis];
	//h_hid_bias = new float[n_hid];

	float *w;
	cudaMallocManaged(&w, n_hid*n_vis*sizeof(float));

	float *wtr;
	cudaMallocManaged(&wtr, n_vis*n_hid*sizeof(float));

	float *vis_bias;
	cudaMallocManaged(&vis_bias, n_vis*sizeof(float));

	float *hid_bias;
	cudaMallocManaged(&hid_bias, n_hid*sizeof(float));

	for(int i = 0; i < n_hid; i++)
		for(int j = 0; j< n_vis; j++){
			w_if >> w[j+i*n_vis];
			//std::cout << "w at index " << j+i*n_vis << "is " << h_w[j+i*n_vis];
		}

	//Reset file stream
	w_if.clear();
	w_if.seekg(std::ios::beg);

	for (int i = 0; i < n_vis; i++)
		for (int j = 0; j < n_hid; j++)
			w_if >> wtr[j*n_vis+i];  //Transpose; column major order writing! Not ideal!

	for(int i = 0; i < n_vis; i++)
		vis_if >> vis_bias[i];

	for(int i = 0; i < n_hid; i++)
		hid_if >> hid_bias[i];

	//Close file streams
	w_if.close();
	vis_if.close();
	hid_if.close();


	//Sample from the trained model using a Gibbs Chain
	std::cout << "Testing the trained model by generating samples..." << std::endl;
	int ret_val = rbm_sample_test(w, wtr, vis_bias, hid_bias, vis_layer, hid_layer,  num_rows_crop, num_cols_crop);
	std::cout << "Testing completed successfully" << std::endl;

	//Cleanup
	//delete [] h_w, h_wtr, h_vis_bias, h_hid_bias;

	if(w != NULL){
		cudaFree(w);
		w = NULL;
	}

	if(wtr != NULL){
		cudaFree(wtr);
		wtr = NULL;
	}

	if(vis_bias != NULL){
		cudaFree(vis_bias);
		vis_bias = NULL;
	}

	if(hid_bias != NULL){
		cudaFree(hid_bias);
		hid_bias = NULL;
	}

	if(vis_layer != NULL){
		cudaFree(vis_layer);
		vis_layer = NULL;
	}

	if(hid_layer != NULL){
		cudaFree(hid_layer);
		hid_layer = NULL;
	}

	return 0;
}
