//Main -- Data Preparation + Model Initialization + Training
//Menna El-Shaer -- February 2018

#include <iostream>
#include <stdlib.h>
#include <fstream>

//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "rbm_functions.hpp"


int main(int argc, char** argv)
{
	//1. Prepare training data
	//Initial parameters
	const int num_frames = 2110;
	const int num_rows_begin = 720;
	const int num_cols_begin = 1280;
	const int num_pixels_begin = num_rows_begin * num_cols_begin;

	//Cropping constants
	const int num_rows_crop = 360;
	const int num_cols_crop = 960;
	const int num_pixels = num_rows_crop * num_cols_crop;
	const int x_crop = 160;
	const int y_crop = 180;
	cv::Rect toCrop = cv::Rect(x_crop, y_crop, num_cols_crop, num_rows_crop);

	double *data = new double[num_frames*num_pixels];

	//Read frames from whole data set, get Mat, transform and add to data array
	cv::String fileName;
	cv::String folderPath = "/home/el-shaer1/cuda-workspace/RBM/xZED/Training_43svo/Frame_";
	for(int frame=1; frame<num_frames; frame++)
	{
		//Read and Crop
		fileName = folderPath + std::to_string(frame) + ".png";
		cv::Mat bgr_image = cv::imread(fileName.c_str(), CV_LOAD_IMAGE_COLOR);
		cv::Mat cropped_ref = bgr_image(toCrop); //creates a reference only
		cv::Mat cropped_temp;
		cropped_ref.copyTo(cropped_temp);

		//Grayscale the cropped image
		cv::Mat cropped;
		cv::cvtColor(cropped_temp, cropped, CV_BGR2GRAY);
		//cv::imshow("GS", cropped);
		//cv::waitKey(0);

		//Normalize (Scale) and convert to double(8 bytes)
		cropped.convertTo(cropped, CV_64F, 1.0/255.0);

		//Copy cropped 2D images into the rows of double data array
		for(int j=0; j<cropped.rows ;j++)
		{
			int i = j * cropped.cols;
			const double *row_ptr = cropped.ptr<double>(j);
			for(int k=0; k<cropped.cols ;k++)
			{
				data[((frame-1)*num_pixels+i)+k] = row_ptr[k];
				//std::cout << "data at index: " << ((frame-1)*num_pixels+i)+k << "is" << data[((frame-1)*num_pixels+i)+k] << std::endl;
			}
		}
	}
	//Zero mean, unit variance?

	//2. Initialize Model
	//RBM Initialization Parameters; will put them in a class object later
	const int n_hid = 400;                    //Number of hidden units
	const int n_randWt = 1;                   //Number of random weight trial sets
	const int n_batches = 10;                 //Number of batches per weight trial
	const int nc = num_frames;                //Number of rows (frames)
	const int n_vis = num_pixels;             //Number of columns (visible units)

	int *shuffle_index = (int *) malloc(nc*sizeof(int));
	double *w = (double *) malloc(n_hid*n_vis*sizeof(double));
	double *vis_bias = (double *) malloc(n_vis*sizeof(double));
	double *hid_bias = (double *) malloc(n_hid*sizeof(double));
	double *w_best = (double *) malloc(n_hid*n_vis*sizeof(double));
	double *vis_bias_best = (double *) malloc(n_vis*sizeof(double));
	double *hid_bias_best = (double *) malloc(n_hid*sizeof(double));
	double *data_mean = (double *) malloc(n_vis*sizeof(double));
	double *err_vec = (double *) malloc(n_vis*sizeof(double));

	std::cout << "Initializing weights..." << std::endl;
	double init_err = weight_init(nc, n_vis, n_vis, data, n_hid, n_randWt, n_batches,
										shuffle_index, w, vis_bias, hid_bias,
										vis_bias_best, hid_bias_best, w_best,
										data_mean, err_vec);
	std::cout << "Initializing weights ended successfully. Best error seen =  " << init_err << std::endl;


	//Display initial weights and biases
	cv::Mat weights_for_display(n_hid, n_vis, CV_64F, w);
	cv::Mat hid_filter(num_rows_crop, num_cols_crop, CV_64F);
	cv::Mat weightsDisplay(num_rows_crop, num_cols_crop, CV_64F);

	for (int i = 0; i < n_hid; i++)
	{
		double maxVal, minVal;
		const double* row_ptr = weights_for_display.ptr<double>(i);

		//Copy row into hid_filter matrix
		for(int j = 0; j < num_rows_crop; j++)
			for(int k = 0; k < num_cols_crop; k++)
				hid_filter.at<double>(j,k) = row_ptr[k+j*num_cols_crop];

		cv::minMaxLoc(hid_filter, &minVal, &maxVal);
		std::cout << "Min = " << minVal << " Max = " << maxVal << std::endl;
		hid_filter.convertTo(weightsDisplay, CV_64F, (1/(maxVal-minVal)), (-minVal/(maxVal-minVal)));
		cv::imshow("Initial Weight Matrix", weightsDisplay);
		cv::waitKey(15);

	}

	//3. Train Model
	const int mv_chain_start = 1;
	const int mv_chain_end = 4;          //Could be 1 or small number
	const double mv_chain_rate = 0.5;    //Exponential Smoothing Rate
	const int mean_field = 1;            //Use mf = 0, use random sampling = 1
	const int greedy_mean_field = 1;     //Use gmf for training = 0 ? yes
	const int max_epochs = 5;            //Maximum number of epochs for training
	const int max_no_improv = 5;         //Converge if this many epochs without ratio improvements
	const double conv_crit = 0.5;
	const double learning_rate = 0.1;
	const double start_momentum = 0.005;
	const double end_momentum = 1;
	const double weight_penalty = 0.005;
	const double sparsity_penalty = 0.1;
	const double sparsity_target = 0.1;

	std::cout << "Training starting..." << std::endl;
    double train_err = rbm_train(nc, n_vis, data, n_vis, n_hid, mv_chain_start, mv_chain_end, mv_chain_rate,
    							mean_field, greedy_mean_field, n_batches, max_epochs, max_no_improv, conv_crit,
    							learning_rate, start_momentum, end_momentum, weight_penalty, sparsity_penalty,
    							sparsity_target, w, vis_bias, hid_bias, shuffle_index, data_mean, err_vec);
    std::cout << "Training completed successfully. Training error for all epochs =  " << train_err << std::endl;

    //4. Display and save final weights and biases
    cv::Mat final_weights_for_display(n_hid, n_vis, CV_64F, w);
    cv::Mat final_hid_filter(num_rows_crop, num_cols_crop, CV_64F);

    for (int i = 0; i < n_hid; i++)
    {
    	double maxVal, minVal;
    	const double *row_ptr = final_weights_for_display.ptr<double>(i);

    	//Copy row into hid_filter matrix
    	for(int j = 0; j < num_rows_crop; j++)
    		for(int k = 0; k < num_cols_crop; k++)
    			final_hid_filter.at<double>(j,k) = row_ptr[k+j*num_cols_crop];

    	cv::minMaxLoc(final_hid_filter, &minVal, &maxVal);
    	std::cout << "Min = " << minVal << " Max = " << maxVal << std::endl;
    	final_hid_filter.convertTo(final_hid_filter, CV_64F, (1/(maxVal-minVal)), (-minVal/(maxVal-minVal)));
    	cv::imshow("Weight Matrix for hidden neuron " + std::to_string(i), final_hid_filter);
    	cv::waitKey(15);
    	final_hid_filter.convertTo(final_hid_filter, CV_64F, 255.0);
    	cv::imwrite(("/home/el-shaer1/cuda-workspace/RBM/xZED/Trained_Parameters/Training_43svo_3/Weight Matrix for hidden neuron " + std::to_string(i) + ".png").c_str(), final_hid_filter);
    }

    std::string w_fileName = "/home/el-shaer1/cuda-workspace/RBM/xZED/Trained_Parameters/Training_43svo_3/weights_" +
    		std::to_string(n_hid) + "hidden_units.txt";
    std::string vis_fileName = "/home/el-shaer1/cuda-workspace/RBM/xZED/Trained_Parameters/Training_43svo_3/vis_bias_" +
    		std::to_string(n_hid) + "hidden_units.txt";
    std::string hid_fileName = "/home/el-shaer1/cuda-workspace/RBM/xZED/Trained_Parameters/Training_43svo_3/hid_bias_" +
    		std::to_string(n_hid) + "hidden_units.txt";

    std::ofstream w_of;
    std::ofstream vis_of;
    std::ofstream hid_of;

    w_of.open(w_fileName);
    vis_of.open(vis_fileName);
    hid_of.open(hid_fileName);

    for(int i=0;i<n_hid;i++)
    	for(int j=0;j<n_vis;j++)
    		w_of << w[j+i*n_vis];

    for(int i=0;i<n_vis;i++)
    	vis_of << vis_bias[i];

    for(int i=0;i<n_hid;i++)
    	hid_of << hid_bias[i];

    w_of.close();
    vis_of.close();
    hid_of.close();


    //5. Cleanup
	delete [] data;

    free(shuffle_index);
    shuffle_index = NULL;
    free(w);
    w = NULL;
    free(vis_bias);
    vis_bias = NULL;
    free(hid_bias);
    hid_bias = NULL;
    free(w_best);
    w_best = NULL;
    free(vis_bias_best);
    vis_bias_best = NULL;
    free(hid_bias_best);
    hid_bias_best = NULL;
    free(data_mean);
    data_mean = NULL;
    free(err_vec);
    err_vec = NULL;

    return 0;
}

