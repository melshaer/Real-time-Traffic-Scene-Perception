//RBM Initialization and Training
//Functions Declaration Header File
//Menna El-Shaer -- January 2017


double weight_init(
		int nc,                 //Number of cases
		int n_vis,              //Number of inputs
		int ncols,              //Number of columns in data
		double *data,           //nc rows by ncols columns of 0-1 input data in first n_vis cols
		int n_hid,              //Number of hidden neurons
		int n_rand,             //Number of random weight sets to try
		int n_batches,          //Number of batches per weight trial
		int *shuffle_index,     //Work vector nc long
		double *w,              //Computed weight matrix, n_hid sets of n_vis weights; max_threads sets
		double *vis_bias,       //Computed input bias vector; max_threads sets
		double *hid_bias,       //Computed hidden bias vector; max_threads sets
		double *vis_bias_best,  //Work vector n_vis long
		double *hid_bias_best,  //Work vector n_hid long
		double *w_best,         //Work vector n_vis * n_hid long
		double *data_mean,      //Work vector n_vis long
		double *err_vec         //Work vector n_vis long
);


double rbm_train(
		int nc,                   //Number of cases in complete dataset
		int ncols,                //Number of columns in data
		double *data,             //nc rows by ncols columns of 0-1 input data in first n_vis cols
		int n_vis,                //Number of visible neurons
		int n_hid,                //Number of hidden neurons
		int mv_chain_start,       //Starting length of Markov chain, generally 1
		int mv_chain_end,         //Ending length of Markov chain, generally 1 or a small number
		double mv_chain_rate,     //Exponential smoothing rate for epochs moving toward n_chain_end
		int mean_field,           //Use mean field instead of random sampling?
		int greedy_mean_field,    //Use mean field for greedy training?
		int n_batches,            //Number of batches per epoch
		int max_epochs,           //Maximum number of epochs
		int max_no_improv,        //Converged if this many epochs with no ratio improvement
		double conv_crit,         //Convergence criterion for max inc / max weight
		double learning_rate,     //Learning rate
		double start_momentum,    //Learning momentum start value
		double end_momentum,      //Learning momentum end value
		double weight_penalty,    //Weight penalty
		double sparsity_penalty,  //Sparsity penalty
		double sparsity_target,   //Sparsity target
		double *w,                //Computed weight matrix, n_hid sets of n_vis weights
		double *vis_bias,         //Computed visible bias vector
		double *hid_bias,         //Computed hidden bias vector
		int *shuffle_index,       //Work vector nc long
		double *data_mean,        //Work vector n_vis long
		double *err_vec           //Work vector n_vis long
);
