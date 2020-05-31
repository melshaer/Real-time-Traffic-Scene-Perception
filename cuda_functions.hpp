//CUDA Functions Declaration Header File
//Menna El-Shaer -- January 2017


enum Error_Codes_t : int { SUCCESS,
	ERROR_INSUFFICIENT_HOST_MEMORY,
	ERROR_INSUFFICIENT_DEVICE_MEMORY,
	ERROR_CUDA_ERROR,
	ERROR_KERNEL_MEMORY_ERROR
};


int cuda_init(
		int nc,                //Number of cases, needed for using shuffle_index for random sampling
		int ncols,             //Number of columns in data (may exceed n_vis)
		int n_vis,             //Number of visible neurons
		int n_hid,             //Number of hidden neurons
		int mean_field,        //Use mean field instead of random sampling?
		int greedy_mean_field, //Use mean field for greedy training?
		int max_batch,         //Max size of any batch
		double *data,          //Input data, nc rows by ncols columns
		double *data_mean,     //Mean of each input, needed for weight sparsity penalty
		double *vis_bias,      //Input bias vector
		double *hid_bias,      //Hidden bias vector
		double *w              //Weight matrix
);

int copy_shuffle_to_device(
		int nc,
		int *shuffle_index
);

int copy_inits_to_device(
		int n_vis,
		int n_hid,
		double *vis_bias,
		double *hid_bias,
		double *w
);

int copy_params_from_device(
		int n_vis,
		int n_hid,
		double *vis_bias,
		double *hid_bias,
		double *w
);

int compute_recon_error(
		int n_vis,
		int nc,
		double *err_vec           //Accumulates MSE for each input; n_vis long
);

int fetch_vis1(
		int istart,               //First case in this batch
		int istop,                //One past last case
		int n_vis,
		int random_offset,        //Starting index in shuffle_index for random sampling
		double *visible1          //If non-NULL, return n_vis * (istop-istart) long vector
);

int vis_to_hid(
		int nc,
		int n_hid,
		double *hidden1,
		double *hidden_act,
		double *hid_on_frac
);

int hid_to_vis(
		int nc,
		int n_vis,
		int random_offset,       //Starting index in shuffle_index for random sampling
		double *visible2         //Work vector n_vis * nc long
);

int hid_to_vis_no_sampling(
		int nc,                  //Number of cases in this batch
		int n_vis,
		double *visible2
);

int vis2_to_hid2(
		int nc,                  //Number of cases in this batch
		int n_hid,               //Number of hidden neurons
		double *hidden2          //Work vector n_hid * (istop-istart) long
);

int sample_hidden2(
		int nc,                  //Number of cases in this batch
		int n_hid,               //Number of hidden neurons
		int random_offset,       //Starting index in shuffle_index for random sampling
		double *hidden_act       //Work vector n_hid * (istop-istart) long
);

int compute_gradients(
		int n ,                  //Number of weights; Not important; just heuristically sets # blocks
		double *len,             //Computed squared length
		double *dot              //Computed dot product
);

int max_inc_weight(
		int n,                   //Number of weights; Not important; just heuristically sets # blocks
		double *max_inc_w ,      //Computed max absolute weight
		const int inc_vs_w       //Compute weight or weight increment?
);

int update_vis_bias(
		int nc,                  //Number of cases in this batch
		int n_vis,               //Number of visible neurons
		double rate,             //Learning rate
		double momentum,         //Learning momentum
		double *vis_bias,        //Visible bias vector, n_vis long
		double *vis_bias_inc     //Visible bias increment vector, carries over from batch to batch, n_vis long
);

int update_hid_bias(
		int nc,                  //Number of cases in this batch
		int n_hid,               //Number of hidden neurons
		double rate,             //Learning rate
		double momentum,         //Learning momentum
		int random_offset,       //Starting index in shuffle_index for random sampling hidden1 if not mean_field
		double sparse_pen,       //Sparsity penalty
		double sparse_targ,      //Sparsity target
		double *hid_bias,        //Hidden bias vector, n_hid long
		double *hid_bias_inc     //Hidden bias increment vector, carries over from batch to batch, n_hid long
);

int update_weights(
		int nc,                  //Number of cases in this batch
		int n_vis,               //Number of visible neurons
		int n_hid,               //Number of hidden neurons
		double rate,             //Learning rate
		double momentum,         //Learning momentum
		double weight_pen,       //Weight penalty
		double sparse_pen,       //Sparsity penalty
		double sparse_targ,      //Sparsity target
		double *w,               //Weight matrix, n_hid sets of n_vis weights
		double *w_inc,           //Weight increment array, carries over from batch to batch, n_hid * n_vis
		double *w_grad           //We'll need gradient for auto update of learning rate; n_hid * n_vis
);

int transpose(
		int n_vis,               //Number of visible neurons
		int n_hid                //Number of hidden neurons
);

void cuda_cleanup();











