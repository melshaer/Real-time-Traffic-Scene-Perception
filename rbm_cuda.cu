//CUDA Functions Definition (inspired from Deep Belief Nets in CUDA C/C++)
//Menna El-Shaer -- February 2018

#include <iostream>
#include <stdio.h>

#include "helper_cuda.h"
#include "cuda_functions.hpp"

//The number of threads MUST be a power of two
//The number of blocks given here is a maximum.  The actual number may be less
#define REDUC_THREADS 256
#define REDUC_BLOCKS 64

//Temporary variables to hold data on conversion to float to device memory
static float *reduc_fdata = NULL;
static float *fdata = NULL;

static cudaDeviceProp deviceProps;

//Define device constant memory variables
__constant__ int d_nc;                //Number of cases (needed for using shuffle_index as random sampler)
__constant__ int d_n_vis;             //Number of inputs (size of visible, bottom layer)
__constant__ int d_n_vis_cols;        //Same, extended to multiple of 128 bytes
__constant__ int d_n_hid;             //Number of hidden neurons
__constant__ int d_n_hid_cols ;       //Same, extended to multiple of 128 bytes
__constant__ int d_mean_field;        //Use mean field instead of random sampling?
__constant__ int d_greedy_mean_field; //Use mean field for greedy training?

//Define host and device pointers
static float *h_data = NULL;                     //Host pointer to float data
__constant__ float *d_data;                      //Data pointer on device

static float *h_data_mean = NULL;                //Host pointer to float data mean
__constant__ float *d_data_mean;                 //Data mean pointer on device

static float *h_vis_bias = NULL;                 //Host pointer to float visible bias
__constant__ float *d_vis_bias;                  //Visible bias pointer on device

static float *h_hid_bias = NULL;                 //Host pointer to float hidden bias vector
__constant__ float *d_hid_bias;                  //Hidden bias vector pointer on device

static float *h_w = NULL;                        //Host pointer to float weight matrix
__constant__ float *d_w;                         //Weight matrix pointer on device

static float *h_wtr = NULL;                      //Host pointer to float transposed weight
__constant__ float *d_wtr;                       //Transposed Weight pointer on device

static int *h_shuffle_index = NULL;              //Host pointer to indexed shuffle vector
__constant__ int *d_shuffle_index;               //Indexed Shuffle Vector pointer on device

static float *h_visible1 = NULL;                 //Host pointer to visible layer1 vector
__constant__ float *d_visible1;                  //Visible Layer1 vector pointer on device

static float *h_visible2 = NULL;	             //Host pointer to visible layer2 vector
__constant__ float *d_visible2;                  //Visible Layer2 vector pointer on device

static float *h_hidden1 = NULL;                  //Host pointer to hidden layer1
__constant__ float *d_hidden1;                   //Hidden Layer1 vector pointer on device

static float *h_hidden2 = NULL;                  //Host pointer to hidden layer2
__constant__ float *d_hidden2 ;                  //Hidden Layer2 vector pointer on device

static float *h_hidden_act = NULL;               //Host pointer to activation value of hidden layer
__constant__ float *d_hidden_act;                //Activation value of hidden layer pointer on device

static float *h_hid_for_sparsity = NULL;         //Host pointer to hidden layer used for sparsity calculations
__constant__ float *d_hid_for_sparsity;          //Hidden layer activation for sparsity pointer on device

static float *h_vis_bias_inc = NULL;             //Host pointer to visible bias increment
__constant__ float *d_vis_bias_inc;              //Visible bias increment pointer on device

static float *h_hid_bias_inc = NULL;             //Host pointer to hidden bias increment
__constant__ float *d_hid_bias_inc;              //Hidden bias increment pointer on device

static float *h_hid_smoothed = NULL;             //Host pointer to smoothed hidden layer
__constant__ float *d_hid_smoothed;              //Smoothed hidden layer pointer on device

static float *h_w_inc = NULL;                    //Host pointer to weight increment values
__constant__ float *d_w_inc;                     //Weight increment values pointer on device

static float *h_w_grad = NULL;                   //Host pointer to weight gradient array
__constant__ float *d_w_grad;                    //Weight gradient array pointer on device

static float *h_prev_w_grad = NULL;              //Host pointer to previous weight gradient array
__constant__ float *d_prev_w_grad;               //Previous weight gradient array pointer on device

static float *h_err_vec = NULL;                  //Host pointer to error vector
__constant__ float *d_err_vec;                   //Error vector pointer on device

static float *h_w_grad_diff = NULL;              //Host pointer to change of weight gradient array
__constant__ float *d_w_grad_diff;               //Change of weight gradient array pointer on device

static float *h_prev_w_grad_diff = NULL;         //Host pointer to the previous change of weight gradient array
__constant__ float *d_prev_w_grad_diff;          //Previous change of weight gradient array pointer on device


//Kernel Functions Declarations
__global__ void recon_error_kernel(int nc);
__global__ void fetch_vis1_kernel(int istart, int random_offset);
__global__ void vis_to_hid_kernel(int nc);
__global__ void hid_to_vis_kernel(int nc, int random_offset);
__global__ void hid_to_vis_no_sampling_kernel(int nc);
__global__ void vis2_to_hid2_kernel(int nc);
__global__ void sample_hidden2_kernel(int nc, int random_offset);
__global__ void compute_gradients_kernel();
__global__ void max_inc_weight_kernel(int inc_vs_w);
__global__ void update_vis_bias_kernel(int nc, float learning_rate, float momentum);
__global__ void update_hid_bias_kernel(int nc, float learning_rate, float momentum, int random_offset, float sparse_penalty, float sparse_target);
__global__ void update_weights_kernel(int nc, float learning_rate, float momentum, float weight_penalty, float sparse_pen , float sparse_targ ) ;
__global__ void transpose_kernel();


//Device Initialization for training
int cuda_init(
   int h_nc ,                // Number of cases, needed for using shuffle_index for random sampling
   int h_ncols ,             // Number of columns in data (may exceed n_vis)
   int h_n_vis ,             // Number of visible neurons
   int h_n_hid ,             // Number of hidden neurons
   int h_mean_field ,        // Use mean field instead of random sampling?
   int h_greedy_mean_field , // Use mean field for greedy training?
   int h_max_batch ,         // Max size of any batch
   double *data ,            // Input data, nc rows by ncols columns
   double *data_mean ,       // Mean of each input, needed for weight sparsity penalty
   double *vis_bias ,        // Input bias vector
   double *hid_bias ,        // Hidden bias vector
   double *w                 // Weight matrix
)
{
   cudaError_t error_id;
   int device = 0;
   int *cuda_device = &device;

   checkCudaErrors(cudaSetDevice(*cuda_device));

   checkCudaErrors(cudaGetDeviceProperties(&deviceProps, *cuda_device));

   //Extend the size of matrices to make sure every row starts on a 128-byte cache-line boundary
   int h_n_vis_cols = (h_n_vis + 31) / 32 * 32 ;
   int h_n_hid_cols = (h_n_hid + 31) / 32 * 32 ;

   //1. Copy constants to device memory
   cudaMemcpyToSymbol(d_nc, &h_nc, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_n_vis, &h_n_vis, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_n_vis_cols, &h_n_vis_cols, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_n_hid, &h_n_hid, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_n_hid_cols, &h_n_hid_cols, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_mean_field, &h_mean_field, sizeof(int), 0, cudaMemcpyHostToDevice);
   cudaMemcpyToSymbol(d_greedy_mean_field, &h_greedy_mean_field, sizeof(int), 0, cudaMemcpyHostToDevice);

   //2. Copy data to device memory and convert to float

   //2.1. Allocate float data on host
   fdata = (float *)malloc(h_nc*h_n_vis*sizeof(float));
   if (fdata == NULL)
	   return ERROR_INSUFFICIENT_HOST_MEMORY;

   //2.2. Allocate h_data space on device
   error_id = cudaMalloc((void **)&h_data, (size_t)(h_nc*h_n_vis*sizeof(float)));
   if (error_id  !=  cudaSuccess)
	   return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   //2.3. Convert data from double to float on host
   for (int i=0 ; i<h_nc ; i++) {
	   for (int j=0 ; j<h_n_vis ; j++)
		   fdata[i*h_n_vis+j] = (float)data[i*h_ncols+j] ;
   }

   //2.4. Copy float data to h_data pointer on device
   error_id = cudaMemcpy(h_data, fdata, h_nc*h_n_vis*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id == cudaSuccess)
	   //2.5. Link h_data pointer to data on device
	   error_id = cudaMemcpyToSymbol(d_data ,&h_data, sizeof(void *), 0, cudaMemcpyHostToDevice);
   if (error_id != cudaSuccess)
   	   return ERROR_CUDA_ERROR;

   free(fdata);
   fdata = NULL; //Reset temporary float variable

   //3. Copy data mean to device memory and convert to float

   //3.1. Allocate float data on host
   fdata = (float *)malloc(h_n_vis*sizeof(float));
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_HOST_MEMORY;

   //3.2. Allocate h_data_mean space on device
   error_id = cudaMalloc((void **)&h_data_mean, (size_t)(h_n_vis*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   //3.3. Convert data mean to float on host
   for (int i=0 ; i<h_n_vis ; i++)
	   fdata[i] = (float)data_mean[i];

   //3.4. Copy float data to h_data_mean pointer on device
   error_id = cudaMemcpy(h_data_mean, fdata, h_n_vis*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id == cudaSuccess)
	   //3.5. Link h_data_mean pointer to data mean on device
	   error_id = cudaMemcpyToSymbol(d_data_mean, &h_data_mean, sizeof(void *), 0, cudaMemcpyHostToDevice) ;
   if (error_id != cudaSuccess)
	   return ERROR_CUDA_ERROR;

   free(fdata);
   fdata = NULL; //Reset temporary float variable

   //4. Copy visible bias to device memory and convert to float

   //4.1. Allocate float data on host
   fdata = (float *)malloc(h_n_vis*sizeof(float));
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_HOST_MEMORY;

   //4.2. Allocate h_vis_bias space on device
   error_id = cudaMalloc((void **)&h_vis_bias, (size_t)(h_n_vis*sizeof(float)));
   if (error_id  !=  cudaSuccess)
	   return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   //4.3. Convert visible bias to float on host
   for (int i=0 ; i<h_n_vis ; i++)
	   fdata[i] = (float)vis_bias[i];

   //4.4. Copy float data to h_vis_bias pointer on device
   error_id = cudaMemcpy(h_vis_bias, fdata, h_n_vis*sizeof(float), cudaMemcpyHostToDevice);
   //4.5. Link h_vis_bias pointer to vis_bias on device
   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol(d_vis_bias, &h_vis_bias, sizeof(void *), 0, cudaMemcpyHostToDevice);
   if (error_id != cudaSuccess)
	   return ERROR_CUDA_ERROR;

   free(fdata);
   fdata = NULL; //Reset temporary float variable

   //5. Copy hidden bias to device memory and convert to float

   //5.1. Allocate float data on host
   fdata = (float *)malloc(h_n_hid*sizeof(float));
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_HOST_MEMORY;

   //5.2. Allocate h_hid_bias space on device
   error_id = cudaMalloc((void **)&h_hid_bias, (size_t)(h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
	   return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   //5.3. Convert hidden bias to float on host
   for (int i=0 ; i<h_n_hid ; i++)
      fdata[i] = (float)hid_bias[i];

   //5.4. Copy float data to h_hid_bias pointer on device
   error_id = cudaMemcpy(h_hid_bias, fdata, h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
   //5.5. Link h_hid_bias pointer to hid_bias on device
   if (error_id == cudaSuccess)
      error_id = cudaMemcpyToSymbol(d_hid_bias, &h_hid_bias, sizeof(void *), 0, cudaMemcpyHostToDevice);
   if (error_id != cudaSuccess)
	   return ERROR_CUDA_ERROR;

   free(fdata);
   fdata = NULL; //Reset temporary float variable

   //6. Copy weight matrix to device memory and convert to float

   //6.1. Allocate float data on host
   fdata = (float *)malloc(h_n_vis_cols*h_n_hid_cols*sizeof(float));
   if (fdata == NULL)
      return ERROR_INSUFFICIENT_HOST_MEMORY;

   //6.2. Allocate h_w and h_wtr space on device
   error_id = cudaMalloc((void **)&h_w, (size_t)(h_n_vis_cols* h_n_hid*sizeof(float))) ;
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   error_id = cudaMalloc((void **)&h_wtr, (size_t)(h_n_vis*h_n_hid_cols*sizeof(float))) ;
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;

  //6.3. Convert weight matrix to float on host and pad zeros to added columns
   int i, j;
   for (j=0 ; j<h_n_hid ; j++) {
      for (i=0 ; i<h_n_vis ; i++)
         fdata[j*h_n_vis_cols+i] = (float) w[j*h_n_vis+i] ;
      for ( ; i<h_n_vis_cols ; i++)
         fdata[j*h_n_vis_cols+i] = 0.0f ;
      }

   //6.4. Copy float data to h_w pointer on device
   error_id = cudaMemcpy(h_w, fdata, h_n_vis_cols*h_n_hid*sizeof(float), cudaMemcpyHostToDevice);

   //6.3. Convert weight matrix to float on host, transpose, and pad zeros to added rows
   if (error_id == cudaSuccess) {
	   for (i=0 ; i<h_n_vis ; i++) {
		   for (j=0 ; j<h_n_hid ; j++)
			   fdata[i*h_n_hid_cols+j] = (float) w[j*h_n_vis+i] ;  //Transpose
		   for ( ; j<h_n_hid_cols ; j++)
			   fdata[i*h_n_hid_cols+j] = 0.0f ;
	   }
	   //6.4. Copy float data to h_wtr pointer on device
	   error_id = cudaMemcpy(h_wtr, fdata, h_n_vis*h_n_hid_cols*sizeof(float), cudaMemcpyHostToDevice);
	         }

   //6.5. Link h_w and h_wtr pointers to w and wtr on device
   if (error_id == cudaSuccess) {
      error_id = cudaMemcpyToSymbol(d_w ,&h_w, sizeof(void *), 0, cudaMemcpyHostToDevice);
      error_id = cudaMemcpyToSymbol(d_wtr, &h_wtr, sizeof(void *), 0, cudaMemcpyHostToDevice);
      }
   if (error_id  !=  cudaSuccess)
   	return ERROR_CUDA_ERROR;

   free(fdata);
   fdata = NULL; //Reset temporary float variable

   //7. Allocate space for work vectors on device and link host pointers

   error_id = cudaMalloc((void **)&h_shuffle_index, (size_t)(h_nc*sizeof(int)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_shuffle_index, &h_shuffle_index, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_visible1, (size_t)(h_max_batch*h_n_vis_cols*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_visible1, &h_visible1, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_visible2, (size_t)(h_max_batch*h_n_vis_cols*sizeof(float)));
   error_id = cudaGetLastError();
   //if (error_id != cudaSuccess)
        //std::cout << cudaGetErrorString(error_id) << std::endl;
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_visible2, &h_visible2, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hidden1, (size_t)(h_max_batch*h_n_hid_cols*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hidden1, &h_hidden1, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hidden2, (size_t)(h_max_batch*h_n_hid_cols*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hidden2, &h_hidden2, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hidden_act, (size_t)(h_max_batch*h_n_hid_cols*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hidden_act, &h_hidden_act, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hid_for_sparsity, (size_t)(h_max_batch*h_n_hid_cols*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hid_for_sparsity,&h_hid_for_sparsity, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_vis_bias_inc, (size_t)(h_n_vis*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_vis_bias_inc, &h_vis_bias_inc, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hid_bias_inc, (size_t)(h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hid_bias_inc, &h_hid_bias_inc, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_hid_smoothed, (size_t)(h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_hid_smoothed, &h_hid_smoothed, sizeof(void * ), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_w_inc, (size_t)(h_n_vis_cols*h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_w_inc, &h_w_inc, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_w_grad, (size_t)(h_n_vis_cols*h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_w_grad, &h_w_grad, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_prev_w_grad, (size_t)(h_n_vis_cols*h_n_hid*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_prev_w_grad, &h_prev_w_grad, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_err_vec, (size_t)(h_n_vis*sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol (d_err_vec, &h_err_vec, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_w_grad_diff, (size_t)(REDUC_BLOCKS * sizeof(float))) ;
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_w_grad_diff, &h_w_grad_diff, sizeof(void *), 0, cudaMemcpyHostToDevice);

   error_id = cudaMalloc((void **)&h_prev_w_grad_diff, (size_t)(REDUC_BLOCKS * sizeof(float)));
   if (error_id  !=  cudaSuccess)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;
   cudaMemcpyToSymbol(d_prev_w_grad_diff, &h_prev_w_grad_diff, sizeof(void *), 0, cudaMemcpyHostToDevice);

   reduc_fdata = (float *)malloc(REDUC_BLOCKS*sizeof(float));
   if (reduc_fdata == NULL)
   	return ERROR_INSUFFICIENT_DEVICE_MEMORY;

   //8. Initialize work vectors on device

   fdata = (float *)malloc(h_n_vis_cols*h_n_hid_cols*sizeof(float));
   if (fdata == NULL)
   	return ERROR_INSUFFICIENT_HOST_MEMORY;

   for (int i=0 ; i<h_n_vis_cols*h_n_hid_cols ; i++)
      fdata[i] = 0.0f;

   error_id = cudaMemcpy(h_vis_bias_inc, fdata, h_n_vis*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy(h_hid_bias_inc, fdata, h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy(h_w_inc, fdata, h_n_vis_cols*h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy(h_w_grad, fdata, h_n_vis_cols*h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
   if (error_id  ==  cudaSuccess)
      error_id = cudaMemcpy(h_prev_w_grad, fdata, h_n_vis_cols*h_n_hid*sizeof(float), cudaMemcpyHostToDevice);

   if (error_id  ==  cudaSuccess) {
	   for (int i=0 ; i<h_n_hid ; i++)
		   fdata[i] = (float) 0.5 ;
	   error_id = cudaMemcpy(h_hid_smoothed, fdata, h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
   }
   if (error_id  !=  cudaSuccess)
	   return ERROR_CUDA_ERROR;

   //9. Reallocate float data variable on host
   int k = h_max_batch * h_n_vis_cols;
   if (h_max_batch * h_n_hid_cols > k)
	   k = h_max_batch * h_n_hid_cols;
   if (h_n_vis_cols * h_n_hid_cols > k)
	   k = h_n_vis_cols * h_n_hid_cols;

   fdata = (float *)realloc(fdata, k*sizeof(float)); //Used for passing parameters back to host
   if (fdata == NULL)
	   return ERROR_INSUFFICIENT_HOST_MEMORY;

   return SUCCESS;
}

int copy_shuffle_to_device(int h_nc, int *shuffle_index)
{
   cudaError_t error_id ;

   error_id = cudaMemcpy(h_shuffle_index, shuffle_index, h_nc*sizeof(int), cudaMemcpyHostToDevice);

   if (error_id  !=  cudaSuccess)
      return ERROR_CUDA_ERROR ;

   return 0 ;
}


int copy_inits_to_device(int h_n_vis, int h_n_hid, double *vis_bias, double *hid_bias, double *w)
{
   cudaError_t error_id;

   int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;
   int h_n_hid_cols = (h_n_hid + 31) / 32 * 32;

   for (int i = 0; i < h_n_vis; i++)
      fdata[i] = (float)vis_bias[i];
   error_id = cudaMemcpy(h_vis_bias, fdata, h_n_vis*sizeof(float), cudaMemcpyHostToDevice);

   if (error_id  ==  cudaSuccess) {
      for (int i = 0; i < h_n_hid; i++)
         fdata[i] = (float)hid_bias[i];
      error_id = cudaMemcpy(h_hid_bias, fdata, h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
      }

   if (error_id  ==  cudaSuccess) {
      for (int j = 0; j < h_n_hid; j++) {
         for (int i = 0; i < h_n_vis; i++)
            fdata[j*h_n_vis_cols+i] = (float)w[j*h_n_vis+i];
         }
      error_id = cudaMemcpy(h_w, fdata, h_n_vis_cols*h_n_hid*sizeof(float), cudaMemcpyHostToDevice);
      }

   if (error_id == cudaSuccess) {
      for (int i = 0; i < h_n_vis; i++) {
         for (int j = 0; j < h_n_hid; j++)
            fdata[i*h_n_hid_cols+j] = (float)w[j*h_n_vis+i];       // Transpose
         }
      error_id = cudaMemcpy(h_wtr, fdata, h_n_vis*h_n_hid_cols*sizeof(float), cudaMemcpyHostToDevice);
      }

   if (error_id  !=  cudaSuccess)
      return ERROR_CUDA_ERROR ;

   return 0 ;
}


//Copies weights and biases from device after training
int copy_params_from_device(int h_n_vis, int h_n_hid, double *vis_bias,	double *hid_bias, double *w)
{
	cudaError_t error_id ;

	int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;

	error_id = cudaMemcpy(fdata, h_w, h_n_hid*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
	for (int ihid=0 ; ihid<h_n_hid ; ihid++) {
		for (int ivis=0 ; ivis<h_n_vis ; ivis++)
			w[ihid*h_n_vis+ivis] = fdata[ihid*h_n_vis_cols+ivis];
	}

	if (error_id == cudaSuccess) {
		error_id = cudaMemcpy(fdata, h_vis_bias, h_n_vis*sizeof(float), cudaMemcpyDeviceToHost);
		for (int ivis=0 ; ivis<h_n_vis ; ivis++)
			vis_bias[ivis] = fdata[ivis];
	}

	if (error_id == cudaSuccess) {
		error_id = cudaMemcpy(fdata, h_hid_bias, h_n_hid*sizeof(float), cudaMemcpyDeviceToHost);
		for (int ihid=0 ; ihid<h_n_hid ; ihid++)
			hid_bias[ihid] = fdata[ihid];
	}

	if (error_id != cudaSuccess) {
		return ERROR_KERNEL_MEMORY_ERROR;
	}

	return SUCCESS;
}

__global__ void recon_error_kernel(int h_nc)
{
   int ivis = blockIdx.x * blockDim.x + threadIdx.x;

   if (ivis >= d_n_vis)
      return;

   float errsum = 0.0f;

/*#if RECON_ERR_XENT
   for (icase=0 ; icase<nc ; icase++) {
      errsum -= d_visible1[icase*d_n_inputs_cols+ivis] * __logf(d_visible2[icase*d_n_inputs_cols+ivis]+0.0000000001f) +
                (1.0f - d_visible1[icase*d_n_inputs_cols+ivis]) * __logf(1.0f-d_visible2[icase*d_n_inputs_cols+ivis]+0.0000000001f) ;
      }
#else*/

   for (int icase=0 ; icase<h_nc ; icase++) {
     float diff = d_visible1[icase*d_n_vis_cols+ivis] - d_visible2[icase*d_n_vis_cols+ivis];
      errsum += diff * diff;
      }
//#endif

   d_err_vec[ivis] = errsum;
}

//Compute Reconstruction Error
int compute_recon_error(int h_n_vis, int h_nc, double *err_vec)
{
   cudaError_t error_id;

   int warpsize = deviceProps.warpSize;      //Threads per warp

   int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize;
   int blocks_per_grid = (h_n_vis + threads_per_block - 1) / threads_per_block;

   recon_error_kernel <<< blocks_per_grid, threads_per_block>>> (h_nc);
   cudaDeviceSynchronize();

   error_id = cudaGetLastError();
   if (error_id != cudaSuccess) {
      printf("%s", cudaGetErrorString(error_id));
      return ERROR_CUDA_ERROR;
	}

   cudaThreadSynchronize();
   error_id = cudaGetLastError();
   if (error_id != cudaSuccess) {
	   return ERROR_CUDA_ERROR;
   }

   //Copy err_vec back to host for checking
   if (err_vec != NULL) {
	   error_id = cudaMemcpy(fdata, h_err_vec, h_n_vis*sizeof(float), cudaMemcpyDeviceToHost);
	   for (int i=0 ; i<h_n_vis ; i++)
		   err_vec[i] = fdata[i];

	   if (error_id != cudaSuccess) {
		   return ERROR_KERNEL_MEMORY_ERROR;
	   }
   }
   return SUCCESS ;
}

//Computes actual visible1 input after it's shuffled and batch selected
__global__ void fetch_vis1_kernel(int istart, int random_offset)
{
   int ivis = blockIdx.x * blockDim.x + threadIdx.x;
   if (ivis >= d_n_vis)
      return;

   int icase = blockIdx.y;

   d_visible1[icase*d_n_vis_cols+ivis] = d_data[d_shuffle_index[istart+icase] * d_n_vis + ivis];

   //If greedy_mean_field is used (value is 0), it samples.
   if (! d_greedy_mean_field) {
      int k = ((unsigned int) (icase * d_n_vis + ivis + random_offset)) % d_nc;
      float frand = (float) d_shuffle_index[k] / (float) d_nc;
      d_visible1[icase*d_n_vis_cols+ivis] = (frand < d_visible1[icase*d_n_vis_cols+ivis])  ?  1.0f : 0.0f;
      }
}

int fetch_vis1(int istart, int istop, int h_n_vis, int random_offset, double *visible1)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
	if (threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;

	block_launch.x = (h_n_vis + threads_per_block - 1) / threads_per_block ;
	block_launch.y = istop - istart ;
	block_launch.z = 1 ;

	fetch_vis1_kernel <<<block_launch, threads_per_block>>> (istart ,random_offset);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	//Copy visible1 back to host for checking
	if (visible1 != NULL) {
		int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;
		error_id = cudaMemcpy(fdata, h_visible1, (istop - istart)*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
		for (int icase=0 ; icase<istop-istart ; icase++) {
			for (int ivis=0 ; ivis<h_n_vis ; ivis++)
				visible1[icase*h_n_vis+ivis] = fdata[icase*h_n_vis_cols+ivis];
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR;
		}
	}
	return SUCCESS;
}

//Computes hidden1 probabilities using visible1
//Copies hidden1 to hidden2 for Markov chain use later
__global__ void vis_to_hid_kernel(int nc)
{
   int ihid = blockIdx.x * blockDim.x + threadIdx.x;
   if (ihid >= d_n_hid)
      return;

   int icase = blockIdx.y;

   float sum = d_hid_bias[ihid];
   for (int ivis=0 ; ivis<d_n_vis ; ivis++)
      sum += d_wtr[ivis*d_n_hid_cols+ihid] * d_visible1[icase*d_n_vis_cols+ivis];

   float act_Q = 1.0f / (1.0f + __expf(-sum));
   d_hidden1[icase*d_n_hid_cols+ihid] = act_Q;
   d_hidden2[icase*d_n_hid_cols+ihid] = act_Q;
   d_hidden_act[icase*d_n_hid_cols+ihid] = act_Q;
   d_hid_for_sparsity[icase*d_n_hid_cols+ihid] = act_Q;
}

int vis_to_hid(int h_nc, int h_n_hid, double *hidden1, double *hidden_act, double *hid_on_frac)
{
   dim3 block_launch;
   cudaError_t error_id;

   int warpsize = deviceProps.warpSize;      //Threads per warp

   int threads_per_block = (h_n_hid + warpsize - 1) / warpsize * warpsize ;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize ;

   block_launch.x = (h_n_hid + threads_per_block - 1) / threads_per_block ;
   block_launch.y = h_nc ;
   block_launch.z = 1 ;

   vis_to_hid_kernel <<< block_launch, threads_per_block >>> (h_nc);
   cudaDeviceSynchronize();

   error_id = cudaGetLastError () ;
   if (error_id != cudaSuccess) {
      return ERROR_CUDA_ERROR;
      }

   //Copy hidden1 back to host for checking
   if (hidden1 != NULL) {
      int h_n_hid_cols = (h_n_hid + 31) / 32 * 32 ;
      error_id = cudaMemcpy(fdata, h_hidden1, h_nc*h_n_hid_cols*sizeof(float), cudaMemcpyDeviceToHost);
      for (int icase=0 ; icase<h_nc ; icase++) {
         for (int ihid=0 ; ihid<h_n_hid ; ihid++)
            hidden1[icase*h_n_hid+ihid] = fdata[icase*h_n_hid_cols+ihid];
         }
      //Copy hidden_act back to host for checking
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy(fdata, h_hidden_act, h_nc*h_n_hid_cols*sizeof(float), cudaMemcpyDeviceToHost);
         for (int icase=0 ; icase<h_nc ; icase++) {
            for (int ihid=0 ; ihid<h_n_hid ; ihid++)
               hidden_act[icase*h_n_hid+ihid] = fdata[icase*h_n_hid_cols+ihid];
            }
         }
      //Copy hid_on_frac back to host for checking
      if (error_id == cudaSuccess) {
         error_id = cudaMemcpy(fdata, h_hid_for_sparsity, h_nc*h_n_hid_cols*sizeof(float), cudaMemcpyDeviceToHost);
         for (int icase=0 ; icase<h_nc ; icase++) {
            for (int ihid=0 ; ihid<h_n_hid ; ihid++)
               hid_on_frac[icase*h_n_hid+ihid] = fdata[icase*h_n_hid_cols+ihid];
            }
         }
      if (error_id != cudaSuccess) {
         return ERROR_KERNEL_MEMORY_ERROR;
         }
      }

   return SUCCESS;
}

//Computes (or samples if mean-field) visible2 using hidden1 probabilities
__global__ void hid_to_vis_kernel(int h_nc, int random_offset)
{
	int ivis = blockIdx.x * blockDim.x + threadIdx.x;
	if (ivis >= d_n_vis)
		return;

	int icase = blockIdx.y;

	float sum = d_vis_bias[ivis] ;
	for (int ihid=0 ; ihid<d_n_hid ; ihid++)
		sum += d_w[ihid*d_n_vis_cols+ivis] * d_hidden_act[icase*d_n_hid_cols+ihid];
	float act_P = 1.0f / (1.0f + __expf(-sum));

	if (d_mean_field)
		d_visible2[icase*d_n_vis_cols+ivis] = act_P;
	else {
		int k = ((unsigned int) (icase * d_n_vis + ivis + random_offset)) % d_nc;
		float frand = (float) d_shuffle_index[k] / (float) d_nc;
		d_visible2[icase*d_n_vis_cols+ivis] = (frand < act_P)  ?  1.0f : 0.0f;
	}

}

int hid_to_vis(int h_nc, int h_n_vis, int random_offset, double *visible2)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
	if (threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;

	block_launch.x = (h_n_vis + threads_per_block - 1) / threads_per_block;
	block_launch.y = h_nc;
	block_launch.z = 1;

	hid_to_vis_kernel <<< block_launch , threads_per_block >>> (h_nc, random_offset);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (visible2 != NULL) {
		int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;
		error_id = cudaMemcpy(fdata, h_visible2, h_nc*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
		for (int icase=0 ; icase<h_nc ; icase++) {
			for (int ivis=0 ; ivis<h_n_vis ; ivis++)
				visible2[icase*h_n_vis+ivis] = fdata[icase*h_n_vis_cols+ivis];
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR;
		}
	}
	return SUCCESS ;
}

//Computes visible2 probabilities using hidden1 (without sampling) and weights
//Used only when initializing weights to reproduce reconstruction error
__global__ void hid_to_vis_no_sampling_kernel(int nc)
{
   int ivis = blockIdx.x * blockDim.x + threadIdx.x;
   if (ivis >= d_n_vis)
      return;

   int icase = blockIdx.y;

   float sum = d_vis_bias[ivis] ;
   for (int ihid=0 ; ihid<d_n_hid ; ihid++)
      sum += d_w[ihid*d_n_vis_cols+ivis] * d_hidden1[icase*d_n_hid_cols+ihid];
   d_visible2[icase*d_n_vis_cols+ivis] = 1.0f / (1.0f + __expf(-sum));

}

int hid_to_vis_no_sampling(int h_nc, int h_n_vis, double *visible2)
{
   dim3 block_launch;
   cudaError_t error_id;

   int warpsize = deviceProps.warpSize;      // Threads per warp

   int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
   if (threads_per_block > 4 * warpsize)
      threads_per_block = 4 * warpsize;

   block_launch.x = (h_n_vis + threads_per_block - 1) / threads_per_block;
   block_launch.y = h_nc;
   block_launch.z = 1;

   hid_to_vis_no_sampling_kernel <<< block_launch, threads_per_block >>> (h_nc);
   cudaDeviceSynchronize();

   error_id = cudaGetLastError();
   if (error_id != cudaSuccess) {
	   return ERROR_CUDA_ERROR;
   }

   //Copy visible2 back to host for checking
   if (visible2 != NULL) {
	   int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;
	   error_id = cudaMemcpy(fdata, h_visible2, h_nc*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
	   for (int icase=0 ; icase<h_nc ; icase++) {
		   for (int ivis=0 ; ivis<h_n_vis ; ivis++)
			   visible2[icase*h_n_vis+ivis] = fdata[icase*h_n_vis_cols+ivis];
	   }
	   if (error_id != cudaSuccess) {
		   return ERROR_KERNEL_MEMORY_ERROR  ;
	   }
   }

   return SUCCESS;
}

//Computes hidden2 probabilities using visibile2
__global__ void vis2_to_hid2_kernel(int h_nc)
{
	int ihid = blockIdx.x * blockDim.x + threadIdx.x;
	if (ihid >= d_n_hid)
		return;

	int icase = blockIdx.y ;

	float sum = d_hid_bias[ihid];
	for (int ivis=0 ; ivis<d_n_vis ; ivis++)
		sum += d_wtr[ivis*d_n_hid_cols+ihid] * d_visible2[icase*d_n_vis_cols+ivis] ;
	d_hidden2[icase*d_n_hid_cols+ihid] = 1.0f / (1.0f + __expf(-sum));
}

int vis2_to_hid2(int h_nc, int h_n_hid, double *hidden2)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_hid + warpsize - 1) / warpsize * warpsize;
	block_launch.x = (h_n_hid + threads_per_block - 1) / threads_per_block;
	block_launch.y = h_nc;
	block_launch.z = 1;

	vis2_to_hid2_kernel <<< block_launch , threads_per_block >>> (h_nc);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (hidden2 != NULL) {
		int h_n_hid_cols = (h_n_hid + 31) / 32 * 32;
		error_id = cudaMemcpy(fdata, h_hidden2, h_nc*h_n_hid_cols*sizeof(float), cudaMemcpyDeviceToHost);
		for (int icase=0 ; icase<h_nc ; icase++) {
			for (int ihid=0 ; ihid<h_n_hid ; ihid++)
				hidden2[icase*h_n_hid+ihid] = fdata[icase*h_n_hid_cols+ihid];
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR ;
		}
	}
	return SUCCESS;
}

//Samples hidden2 into hidden_act
__global__ void sample_hidden2_kernel(int h_nc, int random_offset)
{
   int ihid = blockIdx.x * blockDim.x + threadIdx.x;
   if (ihid >= d_n_hid)
      return;

   int icase = blockIdx.y;

   int k = ((unsigned int) (icase * d_n_hid + ihid + random_offset)) % d_nc;
   float frand = (float) d_shuffle_index[k] / (float) d_nc;

   d_hidden_act[icase*d_n_hid_cols+ihid] = (frand < d_hidden2[icase*d_n_hid_cols+ihid])  ?  1.0f : 0.0f ;
}

int sample_hidden2(int h_nc, int h_n_hid, int random_offset, double *hidden_act)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_hid + warpsize - 1) / warpsize * warpsize;
	block_launch.x = (h_n_hid + threads_per_block - 1) / threads_per_block;
	block_launch.y = h_nc;
	block_launch.z = 1;

	sample_hidden2_kernel <<< block_launch , threads_per_block >>> (h_nc, random_offset);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (hidden_act != NULL) {
		int h_n_hid_cols = (h_n_hid + 31) / 32 * 32;
		error_id = cudaMemcpy(fdata, h_hidden_act, h_nc*h_n_hid_cols*sizeof(float), cudaMemcpyDeviceToHost);
		for (int icase=0 ; icase<h_nc ; icase++) {
			for (int ihid=0 ; ihid<h_n_hid ; ihid++)
				hidden_act[icase*h_n_hid+ihid] = fdata[icase*h_n_hid_cols+ihid];
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR;
		}
	}
	return SUCCESS;
}

//Computes Gradients
//Rows have to be padded with zeros when there are unused elements
__global__ void compute_gradients_kernel()
{
	__shared__ float partial_grad_diff[REDUC_THREADS];
	__shared__ float partial_prev_grad_diff[REDUC_THREADS];

	int index = threadIdx.x;
	int n = d_n_vis_cols * d_n_hid;

	float sum_grad_diff = 0.0f;
	float sum_prev_grad_diff = 0.0f;

	for (int i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
		sum_grad_diff += d_w_grad[i] * d_w_grad[i];
		sum_prev_grad_diff += d_w_grad[i] * d_prev_w_grad[i];
		d_prev_w_grad[i] = d_w_grad[i];
	}

	partial_grad_diff[index] = sum_grad_diff;
	partial_prev_grad_diff[index] = sum_prev_grad_diff;
	__syncthreads();

	for (int i=blockDim.x>>1 ; i ; i>>=1) {
		if (index < i) {
			partial_grad_diff[index] += partial_grad_diff[index+i];
			partial_prev_grad_diff[index] += partial_prev_grad_diff[index+i];
		}
		__syncthreads();
	}

	if (index == 0) {
		d_w_grad_diff[blockIdx.x] = partial_grad_diff[0];
		d_prev_w_grad_diff[blockIdx.x] = partial_prev_grad_diff[0];
	}
}

int compute_gradients(int h_n, double *w_grad_diff, double *prev_w_grad_diff)
{
	cudaError_t error_id;

	int blocks_per_grid = (h_n + REDUC_THREADS - 1) / REDUC_THREADS;
	if (blocks_per_grid > REDUC_BLOCKS)
		blocks_per_grid = REDUC_BLOCKS;

	compute_gradients_kernel <<< blocks_per_grid , REDUC_THREADS >>> ();
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	error_id = cudaMemcpy(reduc_fdata, h_w_grad_diff, blocks_per_grid*sizeof(float), cudaMemcpyDeviceToHost);
	double sum = 0.0;
	for (int i=0 ; i<blocks_per_grid ; i++)
		sum += reduc_fdata[i];
	*w_grad_diff = sum;

	if (error_id == cudaSuccess) {
		error_id = cudaMemcpy(reduc_fdata, h_prev_w_grad_diff, blocks_per_grid*sizeof(float), cudaMemcpyDeviceToHost);
		sum = 0.0;
		for (int i=0 ; i<blocks_per_grid ; i++)
			sum += reduc_fdata[i];
		*prev_w_grad_diff = sum;
	}

	if (error_id != cudaSuccess) {
		return ERROR_KERNEL_MEMORY_ERROR;
	}
	return SUCCESS;
}

//Computes maximum weight or maximum weight increment
//Rows have to be padded with zeros when there are unused elements
__global__ void max_inc_weight_kernel(const int inc_vs_w)
{
	__shared__ float partial_max[REDUC_THREADS];

	int index = threadIdx.x;
	int n = d_n_vis_cols * d_n_hid;                     //Number of elements

	float max_inc_w = 0.0f;

	if (inc_vs_w) {
		for (int i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
			if (fabs(d_w_inc[i]) > max_inc_w)
				max_inc_w = fabs(d_w_inc[i]);
		}
	}
	else {
		for (int i=blockIdx.x*blockDim.x+index ; i<n ; i+=blockDim.x*gridDim.x) {
			if (fabs(d_w[i]) > max_inc_w)
				max_inc_w = fabs(d_w[i]);
		}
	}

	partial_max[index] = max_inc_w;
	__syncthreads();

	for (int i=blockDim.x>>1 ; i ; i>>=1) {
		if (index < i) {
			if (partial_max[index+i] > partial_max[index])
				partial_max[index] = partial_max[index+i];
		}
		__syncthreads();
	}

	if (index == 0)
		d_w_grad_diff[blockIdx.x] = partial_max[0];
}

int max_inc_weight(int h_n,	double *max_inc_w, const int inc_vs_w)
{
	cudaError_t error_id;

	int blocks_per_grid = (h_n + REDUC_THREADS - 1) / REDUC_THREADS;
	if (blocks_per_grid > REDUC_BLOCKS)
		blocks_per_grid = REDUC_BLOCKS;

	max_inc_weight_kernel <<< blocks_per_grid , REDUC_THREADS >>> (inc_vs_w);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	error_id = cudaMemcpy(reduc_fdata, h_w_grad_diff, blocks_per_grid*sizeof(float), cudaMemcpyDeviceToHost);
	*max_inc_w = 0.0;
	for (int i=0 ; i<blocks_per_grid ; i++) {
		if (reduc_fdata[i] > *max_inc_w)
			*max_inc_w = reduc_fdata[i];
	}

	if (error_id != cudaSuccess) {
		return ERROR_KERNEL_MEMORY_ERROR;
	}
	return SUCCESS;
}

//Updates Visible Bias
__global__ void update_vis_bias_kernel(int h_nc, float learning_rate, float momentum)
{
	int ivis = blockIdx.x * blockDim.x + threadIdx.x;

	if (ivis >= d_n_vis)
		return;

	float sum = 0.0f;

	for (int icase=0 ; icase<h_nc ; icase++)
		sum += d_visible1[icase*d_n_vis_cols+ivis] - d_visible2[icase*d_n_vis_cols+ivis];

	d_vis_bias_inc[ivis] = momentum * d_vis_bias_inc[ivis] + learning_rate * sum / h_nc;
	d_vis_bias[ivis] += d_vis_bias_inc[ivis] ;
}

int update_vis_bias(int h_nc, int h_n_vis, double learning_rate,
		            double momentum, double *vis_bias, double *vis_bias_inc)
{
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
	if (threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;
	int blocks_per_grid = (h_n_vis + threads_per_block - 1) / threads_per_block;

	update_vis_bias_kernel <<< blocks_per_grid , threads_per_block >>> (h_nc, (float)learning_rate, (float)momentum);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (vis_bias != NULL) {
		error_id = cudaMemcpy(fdata, h_vis_bias, h_n_vis*sizeof(float), cudaMemcpyDeviceToHost);
		for (int i=0 ; i<h_n_vis ; i++)
			vis_bias[i] = fdata[i];
		if (error_id == cudaSuccess) {
			error_id = cudaMemcpy(fdata, h_vis_bias_inc, h_n_vis*sizeof(float), cudaMemcpyDeviceToHost);
			for (int i=0 ; i<h_n_vis ; i++)
				vis_bias_inc[i] = fdata[i];
		}
	}
	if (error_id != cudaSuccess) {
		return ERROR_KERNEL_MEMORY_ERROR;
	}
	return SUCCESS;
}

//Updates Hidden Bias
__global__ void update_hid_bias_kernel(int h_nc, float learning_rate, float momentum, int random_offset,
		                               float sparse_penalty, float sparse_target)
{
	int ihid = blockIdx.x * blockDim.x + threadIdx.x;

	if (ihid >= d_n_hid)
		return;

	float sum = 0.0f;
	float frac_on = 0.0f;

	if (d_mean_field) { //not using mean-field
		for (int icase=0 ; icase<h_nc ; icase++) {
			sum += d_hidden1[icase*d_n_hid_cols+ihid] - d_hidden2[icase*d_n_hid_cols+ihid];
			frac_on += d_hid_for_sparsity[icase*d_n_hid_cols+ihid] ;
		}
	}
	else { //sample
		for (int icase=0 ; icase<h_nc ; icase++) {
			int k = ((unsigned int)(icase * d_n_hid + ihid + random_offset)) % d_nc;
			float frand = (float)d_shuffle_index[k] / (float)d_nc;
			d_hidden_act[icase*d_n_hid_cols+ihid] = (frand < d_hidden1[icase*d_n_hid_cols+ihid])  ?  1.0f : 0.0f;
			sum += d_hidden_act[icase*d_n_hid_cols+ihid] - d_hidden2[icase*d_n_hid_cols+ihid];
			frac_on += d_hid_for_sparsity[icase*d_n_hid_cols+ihid];
		}
	}

	sum /= h_nc;
	frac_on /= h_nc;

	//Sparsity Calculations
	d_hid_smoothed[ihid] = 0.95f * d_hid_smoothed[ihid] + 0.05f * frac_on;
	sum -= sparse_penalty * (d_hid_smoothed[ihid] - sparse_target);
	if (d_hid_smoothed[ihid] < 0.01)
		sum -= 0.5 * (d_hid_smoothed[ihid] - 0.01);       //0.5 is heuristic
		if (d_hid_smoothed[ihid] > 0.99)
			sum -= 0.5 * (d_hid_smoothed[ihid] - 0.99);

		d_hid_bias_inc[ihid] = momentum * d_hid_bias_inc[ihid] + learning_rate * sum;
		d_hid_bias[ihid] += d_hid_bias_inc[ihid];
}

int update_hid_bias( int h_nc, int h_n_hid,	double learning_rate, double momentum, int random_offset,
		             double sparse_penalty,	double sparse_target, double *hid_bias,	double *hid_bias_inc)
{
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;      //Threads per warp

	int threads_per_block = (h_n_hid + warpsize - 1) / warpsize * warpsize;
	if (threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;
	int blocks_per_grid = (h_n_hid + threads_per_block - 1) / threads_per_block;

	update_hid_bias_kernel <<< blocks_per_grid , threads_per_block >>>
			(h_nc, (float)learning_rate, (float)momentum, random_offset,
					(float)sparse_penalty, (float)sparse_target);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (hid_bias != NULL) {
		error_id = cudaMemcpy(fdata, h_hid_bias, h_n_hid*sizeof(float), cudaMemcpyDeviceToHost);
		for (int i=0 ; i<h_n_hid ; i++)
			hid_bias[i] = fdata[i];
		if (error_id == cudaSuccess) {
			error_id = cudaMemcpy(fdata, h_hid_bias_inc, h_n_hid*sizeof(float), cudaMemcpyDeviceToHost);
			for (int i=0 ; i<h_n_hid ; i++)
				hid_bias_inc[i] = fdata[i];
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR;
		}
	}
	return SUCCESS;
}

//Updates Weights
__global__ void update_weights_kernel(int h_nc, float learning_rate, float momentum, float weight_penalty,
		                              float sparse_penalty, float sparse_target)
{
	int ivis = blockIdx.x * blockDim.x + threadIdx.x;
	if (ivis >= d_n_vis)
		return;

	int ihid = blockIdx.y;

	float sum = 0.0f;
	if (d_mean_field) { //don't use mean-field i.e. no sampling
		for (int icase=0 ; icase<h_nc ; icase++)
			sum +=  d_hidden1[icase*d_n_hid_cols+ihid] * d_visible1[icase*d_n_vis_cols+ivis] -
			d_hidden2[icase*d_n_hid_cols+ihid] * d_visible2[icase*d_n_vis_cols+ivis];
	}
	else { //use sampled hidden probabilities (hidden_act)
		for (int icase=0 ; icase<h_nc ; icase++)
			sum +=  d_hidden_act[icase*d_n_hid_cols+ihid] * d_visible1[icase*d_n_vis_cols+ivis] -
			d_hidden2[icase*d_n_hid_cols+ihid] * d_visible2[icase*d_n_vis_cols+ivis];
	}

	//Sparsity Calculations
	sum /= h_nc;
	sum -= weight_penalty * d_w[ihid*d_n_vis_cols+ivis];
	sum -= d_data_mean[ivis] * sparse_penalty * (d_hid_smoothed[ihid] - sparse_target);

	if (d_hid_smoothed[ihid] < 0.01)
		sum -= d_data_mean[ivis] * 0.5 * (d_hid_smoothed[ihid] - 0.01);       //0.5 is heuristic
		if (d_hid_smoothed[ihid] > 0.99)
			sum -= d_data_mean[ivis] * 0.5 * (d_hid_smoothed[ihid] - 0.99);

		d_w_grad[ihid*d_n_vis_cols+ivis] = sum;
		d_w_inc[ihid*d_n_vis_cols+ivis] = momentum * d_w_inc[ihid*d_n_vis_cols+ivis] + learning_rate * sum;
		d_w[ihid*d_n_vis_cols+ivis] += d_w_inc[ihid*d_n_vis_cols+ivis];
}


int update_weights(int h_nc, int h_n_vis, int h_n_hid, double learning_rate, double momentum,
		           double weight_penalty, double sparse_penalty, double sparse_target, double *w,
		           double *w_inc, double *w_grad)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;  //Threads per warp

	int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
	if (threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;

	block_launch.x = (h_n_vis + threads_per_block - 1) / threads_per_block;
	block_launch.y = h_n_hid;
	block_launch.z = 1;

	update_weights_kernel <<< block_launch , threads_per_block >>>
			(h_nc, (float)learning_rate, (float)momentum, (float)weight_penalty,
			(float)sparse_penalty, (float)sparse_target);
	cudaDeviceSynchronize();

	error_id = cudaGetLastError();
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR;
	}

	if (w != NULL) {
		int h_n_vis_cols = (h_n_vis + 31) / 32 * 32;
		error_id = cudaMemcpy(fdata, h_w, h_n_hid*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
		for (int ihid=0 ; ihid<h_n_hid ; ihid++) {
			for (int ivis=0 ; ivis<h_n_vis ; ivis++)
				w[ihid*h_n_vis+ivis] = fdata[ihid*h_n_vis_cols+ivis];
		}
		if (error_id == cudaSuccess) {
			error_id = cudaMemcpy(fdata, h_w_inc, h_n_hid*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
			for (int ihid=0 ; ihid<h_n_hid ; ihid++) {
				for (int ivis=0 ; ivis<h_n_vis ; ivis++)
					w_inc[ihid*h_n_vis+ivis] = fdata[ihid*h_n_vis_cols+ivis];
			}
		}
		if (error_id == cudaSuccess) {
			error_id = cudaMemcpy(fdata, h_w_grad, h_n_hid*h_n_vis_cols*sizeof(float), cudaMemcpyDeviceToHost);
			for (int ihid=0 ; ihid<h_n_hid ; ihid++) {
				for (int ivis=0 ; ivis<h_n_vis ; ivis++)
					w_grad[ihid*h_n_vis+ivis] = fdata[ihid*h_n_vis_cols+ivis];
			}
		}
		if (error_id != cudaSuccess) {
			return ERROR_KERNEL_MEMORY_ERROR;
		}
	}
	return SUCCESS;
}

//Computes Weight Matrix Transpose
__global__ void transpose_kernel()
{
	int ivis = blockIdx.x * blockDim.x + threadIdx.x;
	if (ivis >= d_n_vis)
		return;

	int ihid = blockIdx.y;

	d_wtr[ivis*d_n_hid_cols+ihid] = d_w[ihid*d_n_vis_cols+ivis];
}


int transpose(int h_n_vis, int h_n_hid)
{
	dim3 block_launch;
	cudaError_t error_id;

	int warpsize = deviceProps.warpSize;  //Threads per warp

	int threads_per_block = (h_n_vis + warpsize - 1) / warpsize * warpsize;
	if(threads_per_block > 4 * warpsize)
		threads_per_block = 4 * warpsize;

	block_launch.x = (h_n_vis + threads_per_block - 1) / threads_per_block;
	block_launch.y = h_n_hid;
	block_launch.z = 1;

	transpose_kernel <<< block_launch , threads_per_block >>> ();
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaPeekAtLastError());
	error_id = cudaGetLastError() ;
	if (error_id != cudaSuccess) {
		return ERROR_CUDA_ERROR ;
	}
	return SUCCESS;
}

void cuda_cleanup()
{
	if (h_data != NULL) {
		cudaFree(h_data);
		h_data = NULL;
	}
	if (h_data_mean != NULL) {
		cudaFree(h_data_mean);
		h_data_mean = NULL;
	}
	if (h_vis_bias != NULL) {
		cudaFree(h_vis_bias);
		h_vis_bias = NULL;
	}
	if (h_hid_bias != NULL) {
		cudaFree(h_hid_bias);
		h_hid_bias = NULL;
	}
	if (h_w != NULL) {
		cudaFree(h_w);
		h_w = NULL;
	}
	if (h_wtr != NULL) {
		cudaFree(h_wtr);
		h_wtr = NULL;
	}
	if (h_shuffle_index != NULL) {
		cudaFree(h_shuffle_index) ;
		h_shuffle_index = NULL;
	}
	if (h_visible1 != NULL) {
		cudaFree(h_visible1) ;
		h_visible1 = NULL;
	}
	if (h_visible2 != NULL) {
		cudaFree(h_visible2) ;
		h_visible2 = NULL;
	}
	if (h_hidden1 != NULL) {
		cudaFree(h_hidden1);
		h_hidden1 = NULL ;
	}
	if (h_hidden2 != NULL) {
		cudaFree(h_hidden2) ;
		h_hidden2 = NULL ;
	}
	if (h_hidden_act != NULL) {
		cudaFree(h_hidden_act);
		h_hidden_act = NULL;
	}
	if (h_vis_bias_inc != NULL) {
		cudaFree(h_vis_bias_inc);
		h_vis_bias_inc = NULL;
	}
	if (h_hid_bias_inc != NULL) {
		cudaFree(h_hid_bias_inc);
		h_hid_bias_inc = NULL;
	}
	if (h_hid_for_sparsity != NULL) {
		cudaFree(h_hid_for_sparsity);
		h_hid_for_sparsity = NULL;
	}
	if (h_hid_smoothed != NULL) {
		cudaFree(h_hid_smoothed);
		h_hid_smoothed = NULL;
	}
	if (h_w_inc != NULL) {
		cudaFree(h_w_inc);
		h_w_inc = NULL;
	}
	if (h_w_grad != NULL) {
		cudaFree(h_w_grad);
		h_w_grad = NULL;
	}
	if (h_prev_w_grad != NULL) {
		cudaFree(h_prev_w_grad);
		h_prev_w_grad = NULL;
	}
	if (h_err_vec != NULL) {
		cudaFree(h_err_vec);
		h_err_vec = NULL;
	}
	if (h_w_grad_diff != NULL) {
		cudaFree(h_w_grad_diff);
		h_w_grad_diff = NULL;
	}
	if (h_prev_w_grad_diff != NULL) {
		cudaFree(h_prev_w_grad_diff) ;
		h_prev_w_grad_diff = NULL;
	}

	if (reduc_fdata != NULL) {
		free(reduc_fdata);
		reduc_fdata = NULL;
	}

	if (fdata != NULL) {
		free(fdata);
		fdata = NULL;
	}

	cudaDeviceReset();
}
