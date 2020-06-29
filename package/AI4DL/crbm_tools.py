
from timeit import default_timer as timer
import numpy as np
from numpy import outer as np_outer
import time
import matplotlib.pyplot as plt
import numexpr as ne
from numexpr import evaluate 
import sys
import os
import random
import inspect
import json
import pickle

class CRBM:
    
    def __init__(self, n_vis, n_hid, n_his,
                 sigma=0.2, monitor_time=True, scale_factor = 0,
                 n_epochs=100, learning_rate=0.005, momentum=0.0,  verbose=1, random_state=1234,
                 patience=3, dtype="Float32"):

        self.n_vis = n_vis
        self.n_hid = n_hid
        self.n_his = n_his
        self.sigma = sigma
        self.monitor_time = monitor_time
        self.scale_factor = scale_factor
        self.dtype = dtype
        self.random_state = random_state

        ## values relevant for the fit method
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.verbose = verbose
        self.num_epochs_trained = 0
        self.patience = patience

        ## Initialize the random value sof the parameters

        if scale_factor == 0:  #scale factor for the random initialization of the weights
            scale_factor = 1./(n_vis * n_his)
            
        if dtype == "Float32":
            dtype = np.float32
        elif dtype == "Float64":
            dtype = np.float64

        np.random.seed(self.random_state)        
        self.W = scale_factor * np.random.normal(0, sigma, [n_hid, n_vis]).astype(dtype)          # vis to hid
        self.A = scale_factor * np.random.normal(0, sigma, [n_vis, n_vis * n_his]).astype(dtype)  # cond to vis
        self.B = scale_factor * np.random.normal(0, sigma, [n_hid, n_vis * n_his]).astype(dtype)  # cond to hid
        self.v_bias    = np.zeros([n_vis, 1]).astype(dtype)
        self.h_bias    = np.zeros([n_hid, 1]).astype(dtype)
        self.dy_v_bias = np.zeros([n_vis, 1]).astype(dtype)
        self.dy_h_bias = np.zeros([n_hid, 1]).astype(dtype) 

        
    def save(self, model_path, model_name):
        """
        Function to save the information contained in the class in a folder.
        The folder will contain 2 `.json` files.
        """
        
        ### Create a folder where to save models (if it does not exist) 
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        ### Create a folder for the current model with name `model_name`
        model_path = os.path.join(model_path, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        else:
            print("The model {} inside folder {} already exists!".format(model_name, model_path))
            return 0

        ### Save all the information to instanciate the same model again
        arguments_init = inspect.signature(CRBM)
        init_ctx = {k:self.__dict__[k] for k in arguments_init.parameters.keys()} 
        
        with open( os.path.join(model_path, "model_initializer") + '.json', 'w') as outfile:
            json.dump(init_ctx, outfile,  ensure_ascii=False)
        
        with open( os.path.join(model_path, "model_dict") + '.pickle', 'wb') as outfile:
            pickle.dump(self.__dict__, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            
    @classmethod
    def load(self, model_path):

        if not os.path.exists(model_path):
            print("The model {} does not exist!".format(model_path))
            return
            
        if not os.path.exists( os.path.join(model_path, "model_initializer.json")):
            print( "File {} is not found.".format(os.path.join(model_path, "model_initializer.json")))
            return
            
        if not os.path.exists( os.path.join(model_path, "model_dict.pickle")):
            print( "File {} is not found.".format(os.path.join(model_path, "model_dict.pickle")))
            return
            
        with open( os.path.join(model_path, "model_initializer") + '.json', 'rb') as file:
            model_initializer  = json.load(file)
    
        with open( os.path.join(model_path, "model_dict") + '.pickle', 'rb') as file:
             model_dict = pickle.load(file)
        
        crbm = CRBM(**model_initializer)
        crbm.__dict__ = model_dict

        return crbm

    def fit(self, X_slices: list):
        """
        Train a CRBM given a list of slices of the time-series.
        
        - x in X_slices should be a np.ndarray of ndim=2 containing features as columns
        """

        if self.momentum >0:
            ctx = { "W_vel" : np.zeros(self.W.shape), 
                    "A_vel" : np.zeros(self.A.shape),
                    "B_vel" : np.zeros(self.B.shape), 
                    "v_bias_vel" : np.zeros(self.v_bias.shape), 
                    "h_bias_vel" : np.zeros(self.h_bias.shape)}

        self.rec_error_per_epoch = []
        self.patience_counter =0

        n_samples = len(X_slices)
        t_start = time.time()
        previous_rec_error = np.inf
        for n in range(self.n_epochs):
            t_iter = time.time()
            err_epoch = 0
            random.shuffle(X_slices)
            for X_curr in X_slices:
                
                dW, dA, dB, dv_bias, dh_bias, rec_error = self.compute_gradient(X_curr)
                grads = (dW, dA, dB, dv_bias, dh_bias)

                if self.momentum >0:
                    #update_weights_sgd_momentum(crbm, grads, learning_rate, ctx, momentum=0.)
                    raise NotImplementedError("Momentum learning To be implemented")
                else:
                    self.update_weights_sgd(grads,  self.learning_rate)
                
                err_epoch += rec_error
                
            self.num_epochs_trained +=1
            current_rec_error = err_epoch/n_samples
            self.rec_error_per_epoch.append(current_rec_error)

            if current_rec_error >= previous_rec_error:
                self.patience_counter +=1
            else:
                self.patience_counter = 0

            if self.verbose ==1:
                print("ep {:04} |  min {:.2f} sec => rec error: {:.4f}".format(n, 
                                                                               int(time.time() - t_start)/60, 
                                                                               current_rec_error))#,end="\r")


            if self.patience_counter == self.patience:
                print(f"Maximum patience={self.patience} achieved")
                break 
                

    def split_vis(self, vis: np.ndarray):

        n_his = vis.shape[0]
        cond  = vis[0:(n_his-1), :].T
        x     = vis[[n_his-1],:].T
        
        assert  self.n_vis == x.shape[0] and self.n_vis == cond.shape[0], \
                "crbm.n_vis = {}, is different from x.shape[0] = {} or cond.shape[0] = {}".format(self.n_vis,
                                                                                                  x.shape[0],
                                                                                                  cond.shape[0])
        return x, cond


    def history_mat_to_vec(self, cond):
        return np.array([cond.flatten('F')]).T


    def sample_hiddens(self, v: np.ndarray, cond: np.ndarray):
        h_mean = sig( np.dot(self.W, v) +  np.dot(self.B, cond) + self.h_bias)
        h_sample = h_mean > np.random.random(h_mean.shape).astype(np.float32)
        return h_sample, h_mean

    def sample_visibles(self, h: np.ndarray, cond: np.ndarray):
        """
        Notice we don't sample or put the sigmoid here since visible units are Gaussian
        """
        v_mean = np.dot(self.W.T, h) + np.dot(self.A, cond) + self.v_bias  
        return v_mean

    def CDK(self, vis,cond, K=1):
        v_pos_mean = vis
        h_pos_sample, h_pos_mean    = self.sample_hiddens(v_pos_mean, cond)
        v_neg_mean                  = self.sample_visibles(h_pos_mean, cond)
        h_neg_sample, h_neg_mean    = self.sample_hiddens(v_neg_mean, cond)

        for i in range(K-1):
            v_neg_mean           = self.sample_visibles(h_neg_mean, cond)
            h_neg, h_neg_mean    = self.sample_hiddens(v_neg_mean, cond)
        
        return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

    def compute_gradient(self, X):
        """
        Computes an approximated gradient of the likelihod (for a given minibatch X) with
        respect to the parameters. 
        """
        vis, cond = self.split_vis(X)
        cond = self.history_mat_to_vec(cond)
            
        v_pos, h_pos, v_neg, h_neg = self.CDK(vis, cond)
        n_obs = vis.shape[1]
        
        # for a sigle observation:  dW = h * v^T - h_hat * v_hat^T
        dW = ( np.dot(h_pos, v_pos.T) - np.dot(h_neg, v_neg.T) ) * (1./n_obs)
        dA = ( np.dot(v_pos, cond.T)  - np.dot(v_neg, cond.T)  ) * (1./n_obs)
        dB = ( np.dot(h_pos, cond.T)  - np.dot(h_neg, cond.T)  ) * (1./n_obs) 
        
        dv_bias = np.mean(v_pos - v_neg, axis=1, keepdims=True)
        dh_bias = np.mean(h_pos - h_neg, axis=1, keepdims=True)
        #print("n_obs:", n_obs)

        rec_error = np.linalg.norm(v_pos - v_neg)
        #print( np.sqrt(np.sum((v_pos - v_neg)**2)))
        
        return dW, dA, dB, dv_bias, dh_bias, rec_error


    def update_weights_sgd(self, grads, learning_rate):
        
        dW, dA, dB, dv_bias, dh_bias = grads #rec_error = compute_gradient(crbm, X)
        self.W += dW * learning_rate
        self.A += dA * learning_rate
        self.B += dB * learning_rate
        
        self.v_bias += dv_bias * learning_rate
        self.h_bias += dh_bias * learning_rate


    def update_weights_sgd_momentum(self, grads, learning_rate, ctx, momentum=0.9):
        
        dW, dA, dB, dv_bias, dh_bias = grads 
        
        ctx["W_vel"]        = ctx["W_vel"]      * self.momentum    +  dW      * learning_rate
        ctx["A_vel"]        = ctx["A_vel"]      * self.momentum    +  dA      * learning_rate
        ctx["B_vel"]        = ctx["B_vel"]      * self.momentum    +  dB      * learning_rate
        ctx["v_bias_vel"]   = ctx["v_bias_vel"] * self.momentum    +  dv_bias * learning_rate
        ctx["h_bias_vel"]   = ctx["h_bias_vel"] * self.momentum    +  dh_bias * learning_rate
        
        self.W += ctx["W_vel"]
        self.A += ctx["A_vel"]
        self.B += ctx["B_vel"]
        
        self.v_bias += ctx["v_bias_vel"]
        self.h_bias += ctx["h_bias_vel"]


    def generate(self, vis, cond_as_vec, n_gibbs=10):
        """ 
        Given initialization(s) of visibles and matching history, generate a sample in the future.
        
            vis:  n_vis * 1 array
                
            cond_as_vec: n_hist * n_vis array
                
            n_gibbs : int
                number of alternating Gibbs steps per iteration
        """
        
        assert cond_as_vec.shape[1] ==1, "cond_as_vec has to be a column vector"
        
        n_seq = vis.shape[0]
        v_pos, h_pos, v_neg, h_neg = CDK_sa(self, vis, cond_as_vec, n_gibbs)
        
        return v_neg

    def persistentCDK(self, presistent_vis, persistent_his, K=1):
        
        vis_sample = presistent_vis
        his = persistent_his
        
        h_pos_sample, h_pos_mean    = self.sample_hiddens(vis_sample, his)
        v_neg_mean                  = self.sample_visibles(h_pos_sample, his)
        h_neg_sample, h_neg_mean    = self.sample_hiddens(v_neg_mean, his)
        
        for i in range(K-1):
            v_neg_mean           = self.sample_visibles(h_neg_sample, his)
            h_neg, h_neg_mean    = self.sample_hiddens(v_neg_mean, his)

        return vis_sample, h_pos_mean , v_neg_mean, h_neg_mean


    def CDK_sa(self, vis,cond, K=1):
        
        v_pos_mean = vis
        h_pos_sample, h_pos_mean    = self.sample_hiddens(v_pos_mean, cond)
        v_neg_mean                  = self.sample_visibles(h_pos_sample, cond)
        h_neg_sample, h_neg_mean    = self.sample_hiddens(v_neg_mean, cond)
        
        for i in range(K-1):
            v_neg_mean           =  self.sample_visibles(h_neg_sample, cond)
            h_neg, h_neg_mean    =  self.sample_hiddens(v_neg_mean, cond)

        return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

    def generate_given_chain(self, persistent_vis, persistent_his_as_vec, n_gibbs=10):
        """ 
        Given initialization: visible vector and current history, generate a sample in the future.
        
            persistent_vis:  (n_vis, 1) array
            
            persistent_his_as_vec: (n_hist, n_vis) array            
            
            n_gibbs : int   
               number of alternating Gibbs steps per iteration
        """
        
        assert persistent_his_as_vec.shape[1] ==1, "his_as_vec has to be a column vector"
        
        n_seq = persistent_vis.shape[0]
        v_pos, h_pos, v_neg, h_neg = self.persistentCDK(persistent_vis, persistent_his_as_vec, n_gibbs)
        
        return v_neg


    def update_history_as_vec(self, current_hist_vec, v_new):
        n_feat = v_new.shape[0]
        current_hist_vec[0:-n_feat] = current_hist_vec[n_feat:] 
        current_hist_vec[-n_feat:] = v_new
        return current_hist_vec


    def generate_n_samples(self, vis, his_as_vec, n_samples, n_gibbs=30, persitent_chain=False):
        """ 
        Given initialization(s) of visibles and matching history, generate a n_samples in the future.
        
        
        persistent_chain=True
            In the positive phase, PCD does not differ from CD training. 
            In the negative phase, however, instead of running a new chain for each parameter update, 
            PCD maintains a single per- sistent chain. The update at time t takes the state of the
            Gibbs chain at time t − 1, performs one round of Gibbs sampling, and uses this state 
            in the negative gradient estimates. 
        
        """
        
        assert his_as_vec.shape[1] ==1, "his_as_vec has to be a column vector"
        samples = []
                
        if persitent_chain is False:
            for i in range(n_samples):
                v_new = self.generate(vis, his_as_vec, n_gibbs)
                self.update_history_as_vec(his_as_vec, v_new)
                samples.append(v_new)    

        else:
            persistent_vis_chain = vis
            persistent_his_as_vec = his_as_vec
            
            for i in range(n_samples):
                persistent_vis_chain = self.generate_given_chain( persistent_vis_chain, persistent_his_as_vec, n_gibbs)
                self.update_history_as_vec(persistent_his_as_vec, persistent_vis_chain)
                samples.append(persistent_vis_chain)

        return samples


    def predict(self, seq):

        n_seq = len(seq)

        if n_seq < self.n_his + 1:
            if self.verbose:
                print(f"Warning, input sequence has len {n_seq} but history has len {self.n_his}")

        activations = np.zeros((n_seq, self.n_hid))
      
        for k in range(self.n_his+1, n_seq):
            X_slice = seq[(k-self.n_his-1):k, :]
            history_vec = self.history_mat_to_vec(X_slice[0:-1,:])
            vis = X_slice[[-1],:].T
            h_preact, h_activations = self.sample_hiddens(vis, history_vec)
            activations[k,:] = h_activations.T

        return activations

def sig(v):
    return ne.evaluate("1/(1 + exp(-v))")

def split_vis(crbm: CRBM, vis: np.ndarray):
    n_his = vis.shape[0]
    cond = vis[0:(n_his-1), :].T
    x = vis[[n_his-1],:].T
    
    assert  crbm.n_vis == x.shape[0] and crbm.n_vis == cond.shape[0], \
            "crbm.n_vis = {}, is different from x.shape[0] = {} or cond.shape[0] = {}".format(crbm.n_vis,
                                                                                              x.shape[0],
                                                                                              cond.shape[0])
    return x, cond

def dynamic_biases_up(crbm: CRBM, cond: np.ndarray):
    crbm.dy_v_bias = np.dot(crbm.A, cond) + crbm.v_bias 
    crbm.dy_h_bias = np.dot(crbm.B, cond) + crbm.h_bias      
        
def hid_means(crbm: CRBM, vis: np.ndarray):
    p = np.dot(crbm.W, vis) + crbm.dy_h_bias
    return sig(p)
    
def vis_means(crbm: CRBM, hid: np.ndarray):   
    p = np.dot(crbm.W.T, hid) + crbm.dy_v_bias
    return sig(p)

def sample_hiddens(crbm: CRBM, v: np.ndarray, cond: np.ndarray):
    h_mean = sig( np.dot(crbm.W, v) +  np.dot(crbm.B, cond) + crbm.h_bias)
    h_sample = h_mean > np.random.random(h_mean.shape).astype(np.float32)
    return h_sample, h_mean

def sample_visibles(crbm: CRBM, h: np.ndarray, cond: np.ndarray):
    """
    Notice we don't sample or put the sigmoid here since visible units are Gaussian
    """
    v_mean = np.dot(crbm.W.T, h) + np.dot(crbm.A, cond) + crbm.v_bias  
    return v_mean

def CDK(crbm, vis,cond, K=1):
    v_pos_mean = vis
    h_pos_sample, h_pos_mean    = sample_hiddens(crbm,  v_pos_mean, cond)
    v_neg_mean                  = sample_visibles(crbm, h_pos_mean, cond)
    h_neg_sample, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)

    for i in range(K-1):
        v_neg_mean           = sample_visibles(crbm, h_neg_mean, cond)
        h_neg, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)
    
    return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

def update_history_as_vec(current_hist_vec, v_new):
    n_feat = v_new.shape[0]
    current_hist_vec[0:-n_feat] = current_hist_vec[n_feat:] 
    current_hist_vec[-n_feat:] = v_new
    return current_hist_vec

def history_mat_to_vec(cond):
    return np.array([cond.flatten('F')]).T


def compute_gradient(crbm, X):
    """
    Computes an approximated gradient of the likelihod (for a given minibatch X) with
    respect to the parameters. 
    """
    vis, cond = split_vis(crbm, X)
    cond = history_mat_to_vec(cond)
        
    v_pos, h_pos, v_neg, h_neg = CDK(crbm, vis, cond)
    n_obs = vis.shape[1]
    
    # for a sigle observation:  dW = h * v^T - h_hat * v_hat^T
    dW = ( np.dot(h_pos, v_pos.T) - np.dot(h_neg, v_neg.T) ) * (1./n_obs)
    dA = ( np.dot(v_pos, cond.T)  - np.dot(v_neg, cond.T)  ) * (1./n_obs)
    dB = ( np.dot(h_pos, cond.T)  - np.dot(h_neg, cond.T)  ) * (1./n_obs) 
    
    dv_bias = np.mean(v_pos - v_neg, axis=1, keepdims=True)
    dh_bias = np.mean(h_pos - h_neg, axis=1, keepdims=True)
    #print("n_obs:", n_obs)

    rec_error = np.linalg.norm(v_pos - v_neg)
    #print( np.sqrt(np.sum((v_pos - v_neg)**2)))
    
    return dW, dA, dB, dv_bias, dh_bias, rec_error

def update_weights_sgd(crbm, grads, learning_rate):
    
    dW, dA, dB, dv_bias, dh_bias = grads #rec_error = compute_gradient(crbm, X)
    crbm.W += dW * learning_rate
    crbm.A += dA * learning_rate
    crbm.B += dB * learning_rate
    
    crbm.v_bias += dv_bias * learning_rate
    crbm.h_bias += dh_bias * learning_rate

def update_weights_sgd_momentum(crbm, grads, learning_rate, ctx, momentum=0.9):
    
    dW, dA, dB, dv_bias, dh_bias = grads 
    
    ctx["W_vel"]        = ctx["W_vel"]      * momentum    +  dW      * learning_rate
    ctx["A_vel"]        = ctx["A_vel"]      * momentum    +  dA      * learning_rate
    ctx["B_vel"]        = ctx["B_vel"]      * momentum    +  dB      * learning_rate
    ctx["v_bias_vel"]   = ctx["v_bias_vel"] * momentum    +  dv_bias * learning_rate
    ctx["h_bias_vel"]   = ctx["h_bias_vel"] * momentum    +  dh_bias * learning_rate
    
    crbm.W += ctx["W_vel"]
    crbm.A += ctx["A_vel"]
    crbm.B += ctx["B_vel"]
    
    crbm.v_bias += ctx["v_bias_vel"]
    crbm.h_bias += ctx["h_bias_vel"]

def get_slice_at_position_k(X, k, n_his):
    """
    Returns a slice of shape  `(n_his + 1)` with the last column beeing the visible
    vector at the current time step `k`.
    """
    assert k > n_his, "Position k = {} is lower than n_his = {}".format(k, n_his)
    assert k <= X.shape[1], "Position k = {} is bigger than number of timesteps of X.shape[1] = {}".format(k, X.shape[0])
    return X[:, (k-(n_his+1)):k]

def build_slices_from_list_of_arrays(list_of_arrays, n_his, n_feat, verbose=0):
    """
    This function creates a list of slices of shape (n_his + 1, n_feat)
    """
    assert list_of_arrays[0].shape[1] == n_feat, "list_of_arrays[0].shape[1]={} but n_feat={}".format( list_of_arrays[0].shape[1], n_feat)
    
    X_slices = []
    
    for m, arr in enumerate(list_of_arrays):
        if arr.shape[0] < n_his + 1:
            if verbose>0:
                print("Sequence {} has length {}".format(m, arr.shape[0])) 
        else:
            for k in range(n_his+1, arr.shape[0] + 1):
                X_slice = arr[(k-n_his-1):k, :]
                if X_slice.shape[0] != n_his+1:
                    if verbose>0:
                        print("error!")
                X_slices.append(X_slice)
                
    return X_slices

def CDK_sa(crbm, vis,cond, K=1):
    
    v_pos_mean = vis
    h_pos_sample, h_pos_mean    = sample_hiddens(crbm,  v_pos_mean, cond)
    v_neg_mean                  = sample_visibles(crbm, h_pos_sample, cond)
    h_neg_sample, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)
    
        
    for i in range(K-1):
        v_neg_mean           = sample_visibles(crbm, h_neg_sample, cond)
        h_neg, h_neg_mean    = sample_hiddens(crbm,  v_neg_mean, cond)

    return v_pos_mean, h_pos_mean , v_neg_mean, h_neg_mean

def generate(crbm, vis, cond_as_vec, n_gibbs=10):
    """ 
    Given initialization(s) of visibles and matching history, generate a sample in the future.
    
        vis:  n_vis * 1 array
            
        cond_as_vec: n_hist * n_vis array
            
        n_gibbs : int
            number of alternating Gibbs steps per iteration
    """
    
    assert cond_as_vec.shape[1] ==1, "cond_as_vec has to be a column vector"
    
    n_seq = vis.shape[0]
    #import pdb; pdb.set_trace()
    #v_pos, h_pos, v_neg, h_neg = CDK(crbm, vis, cond_as_vec, n_gibbs)
    v_pos, h_pos, v_neg, h_neg = CDK_sa(crbm, vis, cond_as_vec, n_gibbs)
    
    return v_neg
    

def generate_n_samples(crbm, vis, cond_as_vec, n_samples, n_gibbs=100):
    """ 
    Given initialization(s) of visibles and matching history, generate a n_samples in the future.
    """
    
    assert cond_as_vec.shape[1] ==1, "cond_as_vec has to be a column vector"
    
    samples = []
    for i in range(n_samples):
        v_new = generate(crbm, vis, cond_as_vec, n_gibbs)
        
        # This should not be here
        #v_new = v_new/np.linalg.norm(v_new)      
        #print("i:", i, "\tv_new:", v_new.T)
        #print("cond_as_vec:", cond_as_vec[-8:].T, "\n\n")
        #v_new[v_new<0] = 0
        
        update_history_as_vec(cond_as_vec, v_new)
        
        samples.append(v_new)

    return samples
