//Restricted Boltzmann Machines Function Definitions
//Menna El-Shaer -- January 2017

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <malloc.h>
#include <float.h>
#include <random>

#include "cuda_functions.hpp"

//Constants for fast random number generation (Taken from "Deep Belief Nets in CUDA C/C++" by Timothy Masters)
#define IA 16807
#define IM 2147483647
#define AM (1.0 / IM)
#define IQ 127773
#define IR 2836


double weight_init(
		int nc ,                 //Number of cases
        int n_vis ,              //Number of inputs
        int ncols ,              //Number of columns in data
        double *data ,           //nc rows by ncols columns of 0-1 input data in first n_vis cols
        int n_hid ,              //Number of hidden neurons
        int n_rand ,             //Number of random weight sets to try
        int n_batches ,          //Number of batches per weight trial
        int *shuffle_index ,     //Work vector nc long
        double *w ,              //Computed weight matrix, n_hid sets of n_vis weights; max_threads sets
        double *vis_bias ,       //Computed visible bias vector; max_threads sets
        double *hid_bias ,       //Computed hidden bias vector; max_threads sets
        double *vis_bias_best ,  //Work vector n_vis long
        double *hid_bias_best ,  //Work vector n_hid long
        double *w_best ,         //Work vector n_vis * nhid long
        double *data_mean,       //Work vector n_vis long
        double *err_vec)         //Work vector n_vis long
{
   double *dptr;

   //1. Find the mean of the data for each input to initialize visible bias terms to reasonable values.
   for (int ivis=0 ; ivis<n_vis ; ivis++)
      data_mean[ivis] = 0.0 ;
   for (int i=0 ; i<nc ; i++) {
      dptr = data + i * ncols ;
      for (int ivis=0 ; ivis<n_vis ; ivis++)
         data_mean[ivis] += dptr[ivis] ;
      }
   for (int ivis=0 ; ivis<n_vis ; ivis++) {
      data_mean[ivis] /= nc ; //divide input mean over cases (average mean data per case)
      if (data_mean[ivis] < 1.e-8)
         data_mean[ivis] = 1.e-8 ;
      if (data_mean[ivis] > 1.0 - 1.e-8)
         data_mean[ivis] = 1.0 - 1.e-8 ; //threshold mean value for every input
      }

   //2. Split a training epoch into batches. We need the maximum batch size
   int n_done = 0;
   int max_batch = 0;
   int n_in_batch;
   for (int ibatch=0 ; ibatch<n_batches ; ibatch++) {
      n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  //Cases left to do / batches left to do
      if (n_in_batch > max_batch)
         max_batch = n_in_batch ;
      n_done += n_in_batch ;
      }

   //3. Initialize CUDA, including sending all data to device
   std::cout << "Initializing CUDA..." << std::endl;
   const int mean_field = 1;
   const int greedy_mean_field = 1;
   int ret_val = cuda_init(nc, ncols, n_vis, n_hid, mean_field, greedy_mean_field, max_batch, data,
                             data_mean, vis_bias, hid_bias, w) ;
   std::cout << "CUDA initialization ended with return value = " << ret_val << std::endl;

   //4.Initialize the shuffle index, which will be used by fetch_vis1() to extract
   //a random batch of cases from the full data set and for random number generation purposes
   for (int icase=0 ; icase<nc ; icase++)
   	shuffle_index[icase] = icase;

   //5. Initialize weight matrix after trying some small weight values with the one with minimum reconstruction error
   //6. Initialize all bias vectors to minus half of the weight sum for rough balance

   double best_err = 1.e40; //Best error seen so far

   //Start trials
   for (int irand=0 ; irand<n_rand ; irand++) {
      double error = 0.0; //Mean squared error for each epoch
      	  	  	  	  	  //Sum of squared differences between input and P[x=1|hidden layer]
      	  	  	  	  	  //or Cross Entropy loss
      if (irand == 0) {
         ret_val = copy_shuffle_to_device(nc, shuffle_index);
        if (ret_val)
           return -1.0 ;
         }

      //Generate the trial weight matrix and bias vectors
      std::default_random_engine generator;
      std::uniform_real_distribution<double> distribution(0.0, 1.0);

      double diff = 4.0 * distribution(generator) / sqrt(sqrt((double)n_vis * n_hid));

      //5.1, 6.1. Calculate hidden biases
      for (int ihid=0 ; ihid<n_hid ; ihid++) {
         double sum = 0.0;
         //Get all visible weights for this hidden neuron
         for (int ivis=0 ; ivis<n_vis ; ivis++) {
            double wt = diff * (distribution(generator) - 0.5);  //This is symmetric with heavy tails
            w[ihid*n_vis+ivis] = wt;                 //Set weight value for this hidden and visible unit
            sum += data_mean[ivis] * wt ;            //We'll need this for this hidden neuron's bias
            }
         hid_bias[ihid] = -sum ;                     //Center the distribution
         }

      //6.2. Calculate visible biases
      for (int ivis=0 ; ivis<n_vis ; ivis++) {
         double sum = 0.0;
         //Get all hidden weights for this visible neuron; using the initialized random weight value
         for (int ihid=0 ; ihid<n_hid ; ihid++)
            sum += w[ihid*n_vis+ivis];              //We'll need this for this visible neuron's bias
         vis_bias[ivis] = log(data_mean[ivis] / (1.0 - data_mean[ivis])) - 0.5 * sum ; //Center the distribution
         }

      //5.2, 6.3. Copy weight and all biases to device memory
      ret_val = copy_inits_to_device(n_vis, n_hid, vis_bias, hid_bias, w);
      if (ret_val)
    	  return -1.0;

      //5.3, 6.4. Evaluate the reconstruction error for this trial weight set
      n_done = 0;
      int istart = 0;
      int istop;

      //An epoch is split into batches of training data
      for (int ibatch=0 ; ibatch<n_batches ; ibatch++) {
         n_in_batch = (nc - n_done) / (n_batches - ibatch);  //Cases left to do / batches left to do
         istop = istart + n_in_batch;

         double *visible1 = (double *) malloc(n_vis*(istop-istart)*sizeof(double));
         double *hidden1 = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
         double *hidden_act = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
         double *hid_for_sparsity = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
         double *visible2 = (double *) malloc(n_vis*(istop-istart)*sizeof(double));

         //Get visible1 from database
         if(visible1 != NULL) {
         ret_val = fetch_vis1(istart, istop, n_vis, 0, visible1);
         if (ret_val) {
            return -1.0 ;
            }
         }

         //Compute hidden1 probability (no sampling)
         //Work vector n_hid * (istop-istart) long
         if(hidden1 != NULL && hidden_act != NULL && hid_for_sparsity != NULL){
         ret_val = vis_to_hid(n_in_batch, n_hid, hidden1, hidden_act, hid_for_sparsity);
         if (ret_val) {
            return -1.0 ;
            }
         }

         //Compute visible2 using hidden1 (no sampling)
         if(visible2 != NULL){
         ret_val = hid_to_vis_no_sampling(n_in_batch, n_vis, visible2);
         if (ret_val) {
            return -1.0 ;
            }
      }

         ret_val = compute_recon_error(n_vis, n_in_batch, err_vec);

         //Accumulate error across the epoch i.e. all batches
         for (int ivis=0 ; ivis<n_vis ; ivis++)
            error += err_vec[ivis];

         free(visible1);
         visible1 = NULL;
         free(hidden1);
         hidden1 = NULL;
         free(hidden_act);
         hidden_act = NULL;
         free(hid_for_sparsity);
         hid_for_sparsity = NULL;
         free(visible2);
         visible2 = NULL;

         istart = istop ; //Next batch
         n_done += n_in_batch ; //One batch done
         } //For all batches

     //5.4, 6.5. If this trial was better, save the best-so-far parameters
      if (error < best_err) {
         best_err = error ; //save error
         for (int ihid=0 ; ihid<n_hid ; ihid++) {
            hid_bias_best[ihid] = hid_bias[ihid]; //save hidden bias
            for (int ivis=0 ; ivis<n_vis ; ivis++)
               w_best[ihid*n_vis+ivis] = w[ihid*n_vis+ivis]; //save weights
            }
          for (int ivis=0 ; ivis<n_vis ; ivis++)
            vis_bias_best[ivis] = vis_bias[ivis]; //save visible bias
       }
      } //Next trial

   //7. Copy the best parameters (in ?_best) into the weights and finish initialization
        //Note: Since the error is stochastic, we cannot expect an exact match with what we will get
        //on the first epoch, which uses the 'best' weights. But they should usually be close.
   for (int ihid=0 ; ihid<n_hid ; ihid++) {
      hid_bias[ihid] = hid_bias_best[ihid];  //copy best hidden bias
      for (int ivis=0 ; ivis<n_vis ; ivis++)
         w[ihid*n_vis+ivis] = w_best[ihid*n_vis+ivis]; //copy best weights
      }
   for (int ivis=0 ; ivis<n_vis ; ivis++)
      vis_bias[ivis] = vis_bias_best[ivis]; //copy best visible bias

   cuda_cleanup();

   return best_err / (nc * n_vis);
}


double rbm_train(
		int nc,                           //Number of cases in complete data set
        int ncols,                        //Number of columns in data
        double *data,                     //nc rows by ncols columns of 0-1 input data in first n_vis cols
        int n_vis,                        //Number of visible neurons
        int n_hid,                        //Number of hidden neurons
        int mv_chain_start,               //Starting length of Markov chain, generally 1
        int mv_chain_end,                 //Ending length of Markov chain, generally 1 or a small number
        double mv_chain_rate,             //Exponential smoothing rate for epochs moving toward mv_chain_end
        int mean_field,                   //Use mean field instead of random sampling?
        int greedy_mean_field,            //Use mean field for greedy training?
        int n_batches,                    //Number of batches per epoch
        int max_epochs,                   //Maximum number of epochs
        int max_no_improv,                //Converged if this many epochs with no ratio improvement
        double conv_crit,                 //Convergence criterion for max inc / max weight
        double learning_rate,             //Learning rate
        double start_momentum,            //Learning momentum start value
        double end_momentum,              //Learning momentum end value
        double weight_penalty,            //Weight penalty
        double sparsity_penalty,          //Sparsity penalty
        double sparsity_target,           //Sparsity target
        double *w,                        //Computed weight matrix, n_hid sets of n_vis weights
        double *vis_bias,                 //Computed input bias vector
        double *hid_bias,                 //Computed hidden bias vector
        int *shuffle_index,               //Work vector nc long
        double *data_mean,                //Work vector n_vis long
        double *err_vec)                  //Work vector n_vis long
{
	int randnum = 1;                      //Used in the random offset generator for mean-field sampling
	                                      //and hidden2 computations (mainly)
	const bool wts_init = true;           //Did you run the function to initialize weights?

	if(!wts_init){
		//Find the mean of each input for sparsity penalty on weights
		//The 'data' array MUST have n_vis columns on the device to avoid wasting memory.
		//But it will be called with data having ncols columns, with the required data
		//being in the first n_vis columns
		for (int ivis=0 ; ivis<n_vis ; ivis++)
			data_mean[ivis] = 0.0;

		for (int icase=0 ; icase<nc ; icase++) {
			for (int ivis=0 ; ivis<n_vis ; ivis++)
				data_mean[ivis] += data[icase*ncols+ivis];
		}

		for (int ivis=0 ; ivis<n_vis ; ivis++) {
			data_mean[ivis] /= nc;
			if (data_mean[ivis] < 1.e-8)
				data_mean[ivis] = 1.e-8;
			if (data_mean[ivis] > 1.0 - 1.e-8)
				data_mean[ivis] = 1.0 - 1.e-8;
		}

		//Initialize the shuffle index, which will be used by fetch_vis1() to extract
		//a random batch of cases from the full data set and also for random number generation
		for (int icase=0 ; icase<nc ; icase++)
			shuffle_index[icase] = icase;
	}

	//Split a training epoch into batches. We need the maximum batch size
	int n_done = 0;
	int max_batch = 0;
	int n_in_batch;
	for (int ibatch=0 ; ibatch<n_batches ; ibatch++) {
		n_in_batch = (nc - n_done) / (n_batches - ibatch) ;  //Cases left to do / batches left to do
		if (n_in_batch > max_batch)
			max_batch = n_in_batch ;
		n_done += n_in_batch ;
	}

	//Initialize CUDA, including sending all data to device
	std::cout << "Initializing CUDA for training..." << std::endl;
	int ret_val = cuda_init(nc, ncols, n_vis, n_hid, mean_field, greedy_mean_field, max_batch, data,
			data_mean, vis_bias, hid_bias, w) ;
	std::cout << "CUDA initialization for training ended with return value = " << ret_val << std::endl;

   //Start Training
   std::cout << "Training Started... " << std::endl;

   double momentum = start_momentum;
   double chain_length = mv_chain_start;

   int random_offset = 0;                 //For random sampling the visible units when using greedy mean-field and
                                          //Starting index in shuffle_index for random sampling

   double error = 0.0;                    //Accumulates reconstruction error across epoch (sum of all batches)
   double best_crit;
   int num_no_improv = 0;                 //Counts failure of ratio to improve

   int inc_vs_w;                          //1 or 0 ? Compute w_inc : Compute w
   double max_w;                          //For storing temp maximum weight
   double max_w_inc;                      //For storing temp maximum weight increment
   double max_weight_inc = 0.0;           //For testing convergence: increment relative to largest magnitude weight


   //1. Start Epoch Loop: Each epoch is a complete pass through all training data
   for(int iepoch=0 ; iepoch<max_epochs ; iepoch++) {

	   std::cout << "Starting epoch number  " << iepoch << std::endl;

	   //1.1. Shuffle the data so that if it has serial correlation, similar cases do not end up
	   //in the same batch.  It's also nice to vary the contents of each batch,
	   //epoch to epoch, for more diverse averaging.
	   int num_remaining = nc; //Number remaining to be shuffled

	   std::default_random_engine generator;
	   std::uniform_real_distribution<double> distribution(0.0, 1.0);

	   //Fisher-Yates Shuffle O(n)
	   while (num_remaining > 1) {                  //While at least 2 left to shuffle
		   int j = (int)(distribution(generator) * num_remaining);
		   if (j >= num_remaining)
			   j = num_remaining - 1;
		   random_offset = shuffle_index[--num_remaining];
		   shuffle_index[num_remaining] = shuffle_index[j];
		   shuffle_index[j] = random_offset;
	   }

	  //2. Copy shuffle to device memory
      ret_val = copy_shuffle_to_device(nc , shuffle_index);
      if (ret_val)
         return -1.0;

      //3. Start batch loop
      int istart = 0;                                              //Batch start = training data start
      n_done = 0;                                                  //Number of training cases done in this epoch so far

      double *vis_bias_inc = (double *) malloc(n_vis*sizeof(double)); //Visible Bias Inc; carries over from batch to batch
      double *hid_bias_inc = (double *) malloc(n_hid*sizeof(double)); //Hidden Bias Inc; carries over from batch to batch
      double *w_inc = (double *) malloc(n_hid*n_vis*sizeof(double));  //Weight Bias Inc; carries over from batch to batch
      double *w_grad = (double *) malloc(n_hid*n_vis*sizeof(double)); //Weight Gradient; for updating the learning rate

      double w_grad_diff_current, w_grad_diff_previous, prev_w_grad_diff;
      double smoothed_current, smoothed_diff, smoothed_ratio;

      //An epoch is split into batches of training data
      for(int ibatch=0 ; ibatch<n_batches ; ibatch++) {
    	  std::cout << "Starting batch number " << ibatch << std::endl;
    	  n_in_batch = (nc - n_done) / (n_batches - ibatch);  //Cases left to do / batches left to do
    	  int istop = istart + n_in_batch;                    //Stop just before this index

    	  //Host Variables
    	  double *visible1 = (double *) malloc(n_vis*(istop-istart)*sizeof(double));
    	  double *hidden1 = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
    	  double *hidden_act = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
    	  double *hid_for_sparsity = (double *) malloc(n_hid*(istop-istart)*sizeof(double));
    	  double *visible2 = (double *) malloc(n_vis*(istop-istart)*sizeof(double));
    	  double *hidden2 = (double *) malloc(n_hid*(istop-istart)*sizeof(double));

    	  //Random Offset Generator
    	  if (! greedy_mean_field) {
    		  random_offset = randnum / IQ ;
    		  randnum = IA * (randnum - random_offset * IQ) - IR * random_offset ;
    		  if (randnum < 0)
    			  randnum += IM ;
    	  }

    	  //3.1. Get visible unit values from data
    	  ret_val = fetch_vis1(istart, istop, n_vis, random_offset, visible1);
    	  if (ret_val) {
    		  return -1.0;
    	  }

    	  //3.2. Compute hidden1 probability (no sampling); also copy to hidden2 for MC chain
    	  ret_val = vis_to_hid(n_in_batch, n_hid, hidden1, hidden_act, hid_for_sparsity);
    	  if (ret_val)
    		  return -1.0;

    	  //4. Start Markov Chain Loop
    	  std::cout << "Markov Chain Started... " << std::endl;
    	  for (int ichain=0 ; ichain<(int)(chain_length+0.5)  ; ichain++) {
    		  std::cout << "Starting with chain number  " << ichain << std::endl;

    		  //4.1.Sample hidden2 into hidden_act
    		  int k = randnum / IQ;
    		  randnum = IA * (randnum - k * IQ) - IR * k;
    		  if (randnum < 0)
    			  randnum += IM;
    		  ret_val = sample_hidden2(n_in_batch, n_hid, randnum, hidden_act) ;
    		  if (ret_val)
    			  return -1.0 ;

    		  //4.2. Use hidden_act to get visible2, sampling visible2 if not mean_field
    		  if (! mean_field) {
    			  k = randnum / IQ;
    			  randnum = IA * (randnum - k * IQ) - IR * k;
    			  if (randnum < 0)
    				  randnum += IM;
    		  }
    		  ret_val = hid_to_vis(n_in_batch, n_vis, randnum, visible2);
    		  if (ret_val)
    			  return -1.0;

    		  //4.3. Compute reconstruction error for the first chain
    		  if (ichain == 0) {
    			  ret_val = compute_recon_error(n_vis, n_in_batch, err_vec);
    			  if (ret_val)
    				  return -1.0;
    		  }

    		  //4.4. Use visible2 (which is probabilities or samples per mean_field)
    		  //to get hidden2 probabilities (no sampling of hidden2)
    		  ret_val = vis2_to_hid2(n_in_batch, n_hid, hidden2);
    		  if (ret_val)
    			  return -1.0;
    	  } //Markov chain Loop

    	  std::cout << "Markov Chain Ended Successfully " << std::endl;

    	  //5. Update parameters, accumulate error, and keep track of max weight increment for convergence test later
    	  std::cout << "Updating parameters now... " << std::endl;

    	  //5.1. Update visible bias
    	  ret_val = update_vis_bias(n_in_batch, n_vis, learning_rate, momentum, vis_bias, vis_bias_inc);
    	  if (ret_val)
    		  return -1.0;

    	  //5.2. Update hidden bias
    	  //We'll need randnum (if not mean_field) to sample hidden1 into hidden_act
    	  if(! mean_field) {
    		  int k = randnum / IQ;
    		  randnum = IA * (randnum - k * IQ) - IR * k;
    		  if (randnum < 0)
    			  randnum += IM;
    	  }
    	  ret_val = update_hid_bias(n_in_batch, n_hid, learning_rate, momentum,
    			  randnum, sparsity_penalty, sparsity_target, hid_bias, hid_bias_inc);
    	  if (ret_val)
    		  return -1.0;

    	  //5.3. Update weights
    	  ret_val = update_weights(n_in_batch, n_vis, n_hid, learning_rate, momentum, weight_penalty,
    			  sparsity_penalty, sparsity_target, w, w_inc, w_grad);
    	  if (ret_val)
    		  return -1.0;

    	  ret_val = transpose(n_vis, n_hid);
    	  if (ret_val)
    		  return -1.0;

    	  //5.4. Accumulate error across epoch (all batches)
    	  for (int ivis=0 ; ivis<n_vis ; ivis++)
    		  error += err_vec[ivis]; //Add current batch error to previous batch for a single epoch error

    	  //5.5. Keep track of maximum weight increments. This function uses parallel reduction to find maximums
    	  inc_vs_w = 1; //compute max_inc
    	  ret_val = max_inc_weight(n_vis*n_hid, &max_w_inc, inc_vs_w);
    	  if (ret_val)
    		  return -1.0;

    	  if (max_w_inc > max_weight_inc)
    		  max_weight_inc = max_w_inc;

    	  //6. Compute gradients (and previous) lengths and dot products for dynamic updating of learning rate
    	  if (iepoch == 0  &&  ibatch == 0) {            //No previous gradient yet
    		  ret_val = compute_gradients(n_vis*n_hid, &w_grad_diff_previous, &prev_w_grad_diff);
    		  if (ret_val)
    			  return -1.0;
    		  //For user display only
    		  smoothed_current = sqrt(w_grad_diff_previous / (n_hid * n_vis));
    		  smoothed_diff = 0.0;
    	  }
    	  else {
    		  ret_val = compute_gradients(n_vis*n_hid, &w_grad_diff_current, &prev_w_grad_diff);
    		  if (ret_val)
    			  return -1.0;

    		  prev_w_grad_diff /= sqrt(w_grad_diff_current * w_grad_diff_previous);
    		  w_grad_diff_previous = w_grad_diff_current;

    		  if (prev_w_grad_diff > 0.5)                //Heuristic threshold (0.5)
    			  learning_rate *= 1.2;
    		  else if (prev_w_grad_diff > 0.3)
    			  learning_rate *= 1.1;
    		  else if (prev_w_grad_diff < -0.5)
    			  learning_rate /= 1.2;
    		  else if (prev_w_grad_diff < -0.3)
    			  learning_rate /= 1.1;
    		  if (learning_rate > 1.0)
    			  learning_rate = 1.0;
    		  if (learning_rate < 0.001)
    			  learning_rate = 0.001;

    		  if (fabs(prev_w_grad_diff) > 0.3)
    			  momentum /= 1.5;

    		  smoothed_current = 0.99 * smoothed_current + 0.01 * sqrt(w_grad_diff_current / (n_hid * n_vis));
    		  smoothed_diff = 0.9 * smoothed_diff + 0.1 * prev_w_grad_diff;
    	  }

    	  //Free host variables
    	  free(visible1);
    	  visible1 = NULL;
    	  free(hidden1);
    	  hidden1 = NULL;
    	  free(hidden_act);
    	  hidden_act = NULL;
    	  free(hid_for_sparsity);
    	  hid_for_sparsity = NULL;
    	  free(visible2);
    	  visible2 = NULL;
    	  free(hidden2);
    	  hidden2 = NULL;

    	  n_done += n_in_batch; //One batch done
    	  istart = istop;       //Continue
      } //Batch done

      free(vis_bias_inc);
      vis_bias_inc = NULL;
      free(hid_bias_inc);
      hid_bias_inc = NULL;
      free(w_inc);
      w_inc = NULL;
      free(w_grad);
      w_grad = NULL;

      std::cout << "All batches completed successfully" << std::endl;

      error /= nc * n_vis;

      //7. Test for convergence: largest gradient across epoch relative to largest magnitude weight
      inc_vs_w = 0; //compute w
      ret_val = max_inc_weight(n_vis * n_hid, &max_w, inc_vs_w);
      if (ret_val)
         return -1.0;

      //7.1. Compute ratio of maximum weight increments from all batches in current epoch to the largest
      //weight in the current epoch
      if (max_weight_inc / max_w < conv_crit)
    	  break; //Finish Training i.e. Convergence Reached! i.e. weights not changing much anymore

      //7.2. Define a second-best convergence criteria that is related to the weight increments in the data
      //Too many failures to improve when we get near convergence, the stochastic nature of the gradient calculation
      //causes the update to wander aimlessly
      if (iepoch == 0  ||  max_weight_inc / max_w < best_crit) {
    	  best_crit = max_weight_inc / max_w;
    	  num_no_improv = 0;                     //Number of epochs with no improvement
      }
      else {
    	  ++num_no_improv;
    	  if (num_no_improv > max_no_improv)
    		  break ;                            //Give Up :(
      }

      //7.3. Adjust momentum and markov chain length values for the next epoch
      momentum = 0.99 * momentum + 0.01 * end_momentum;
      chain_length = (1.0 - mv_chain_rate) * chain_length + mv_chain_rate * mv_chain_end; //exponential smoothing

      if (iepoch == 0)
         smoothed_ratio = max_weight_inc / max_w;
      else
         smoothed_ratio = 0.9 * smoothed_ratio + 0.1 * max_weight_inc / max_w;

      //7.4. Adjust the learning rate for the next epoch to prevent wild gyrations when near convergence
      if (num_no_improv > 50  &&  learning_rate > 0.03)
         learning_rate = 0.03;
      if (num_no_improv > 100  &&  learning_rate > 0.02)
         learning_rate = 0.02;
      if (num_no_improv > 150  &&  learning_rate > 0.01)
         learning_rate = 0.01;
      if (num_no_improv > 200  &&  learning_rate > 0.005)
         learning_rate = 0.005;
      if (num_no_improv > 250  &&  learning_rate > 0.002)
         learning_rate = 0.002;

      } //Epoch Loop

      std::cout << "All epochs completed successfully" << std::endl;

      ret_val = copy_params_from_device(n_vis, n_hid, vis_bias, hid_bias, w);
      if (ret_val)
    	  return -1.0;

      cuda_cleanup();

      return error;
}

