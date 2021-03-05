# -*- coding: iso-8859-1 -*-

"""
Using Tensorflow to train GP kernels.

"""

import time, copy, os
# from __future__ import print_function
# from __future__ import absolute_import
# from __future__ import division
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
# from chaospy.distributions.sampler.sequences.hammersley import create_hammersley_samples
# from scipy.special import erfinv
from scipy.linalg import cho_factor, cho_solve
import progressbar
import time
import warnings
warnings.filterwarnings("ignore")


class tfgp_trainer:
    
    def __init__(self, X, y, verbose=0):
        '''
        tfgp_trainer class to train GP kernels 
        
        Arguments
        ---------
        X : ndarray, shape (p,n)
            Array of p observed input data samples.
        y : ndarray, shape (p,1)
            Array of p observed objective function values.
        '''
            
        self.set_data(X, y)
        self.jitter = 1.e-4                   # increase for numerical stability
        self.load_TFGP_path = None            # change to string to load a saved 
        self.dtype = tf.float64               # precision for tensorflow
        self.Verbose = verbose                # print stuff for debugging
        self.tol = 1e-15                      # tolerance for which the iterations will break - i.e. log likelihood was converged. 
        
    def set_data(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.dim = np.shape(X)[1]
        if len(self.y.shape) == 1:
              self.y = np.transpose(np.array(self.y, ndmin=2))
        else: 
            print('WARNING: objective is not a simple scalar... y.shape is ', y.shape)     
    
  
    def logguard(self,x):
        """
        Function to solve the senstivity of the log function in TF
        """
        if x is not None: 
            return tf.cast(tf.math.log(x),dtype=tf.float64)
        elif x<=0 and x<=-0:
            return tf.cast(-10,dtype=tf.float64)
        else: 
            return tf.cast(1e-8,dtype=tf.float64)
                       
        
        
    def dist_square(self, X1, X2, precision_matrix=None):
        """
        Function to compute the squared distance between two arrays, including scaling matrix (p)
        
        |(x1-x2)^T*precision_matrix*(x1-x2)| = |x1^T*precision_matrix*x1| + |x2^T*precision_matrix*x2| 
                                             - |x1^T*precision_matrix*x2| - |x2^T*precision_matrix*x1|
        
        Args:
        -----
        X1, X2:  ndarray, shape (p,1)
                    
        precision_matrix: 
                         Vector of lengthscales or 
                         Full Matrix with correlations and length scales 
                         
                         * recall - precision is inverse covariance matrix if it exists.
                         * note - precision must be symmetric
                   
            
        Returns:
        --------

            dist_square: ndarray
                        |(x1-x2)^T*precision_matrix*(x1-x2)|
        
        """
        
        size1 = int(X1.shape[0]); size2 = int(X2.shape[0])
                    
        if type(precision_matrix) is type(None): 
        # no length scale
            X1_sum_sq = tf.reshape(tf.reduce_mean(tf.square(X1),axis=1), (size1, 1))
            X2_sum_sq = tf.reshape(tf.reduce_mean(tf.square(X2),axis=1), (1, size2))
            cross_prod = tf.matmul(X1, tf.transpose(X2))
            dist_square = X1_sum_sq - 2. * cross_prod + X2_sum_sq
                    
        else: 
        # assume precision matrix
            if len(precision_matrix.shape) == 1: 
                # precision_matrix is a vector of (inverse square) length scales
                p = tf.diag(precision_matrix)
            else:
               # full matrix with correlations and length scales
                p = precision_matrix 

            p_x1 = tf.matmul(X1,p)
            p_x2 = tf.matmul(X2,p)
                                 
            X1_sum_sq = tf.reshape(tf.reduce_sum(X1 * p_x1, axis=1), (size1,1))
            X2_sum_sq = tf.reshape(tf.reduce_sum(X2 * p_x2, axis=1), (1,size2))

            dist_square0 = -2 * tf.matmul(X1, tf.transpose(p_x2)) # works for symmetric precision
            dist_square = tf.abs(X1_sum_sq + X2_sum_sq + dist_square0)
        
        if self.Verbose:
            with tf.compat.v1.Session() as sess:
                print(f'(dist_square, X1_sum_sq, X2_sum_sq, dist_square0):{sess.run(dist_square, X1_sum_sq, X2_sum_sq, dist_square0)}')
            
        return dist_square 


        
        
    def kernel_rbf(self, X1, X2, precision_matrix=None):
        # RBF kernel function 
        # https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        
        dist_sq = self.dist_square(X1, X2, precision_matrix)
        K = tf.exp(-0.5 * dist_sq) 
        
        if self.Verbose:
            print(f'Kernel Matrix:{K}')

        return K 

    

###############################################################################################    
    def kernel_matern(self, X1, X2, precision_matrix=None, nu=1.5):    
        """
        Matern kernel function 
        https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        
        Note: The smaller nu, the less smooth the approximated function is. 
        For nu=inf, the kernel becomes equivalent to the RBF kernel and 
        for nu=0.5 to the absolute exponential kernel. 
        Important intermediate values are nu=1.5 (once differentiable functions) and nu=2.5 
        (twice differentiable functions). 
        See Rasmussen and Williams 2006, pp84 for details regarding the different variants of the Matern kernel.
        """
    
        dist_sq =  self.dist_square(X1, X2, precision_matrix)
        dist = tf.cast(tf.sqrt(dist_sq + 1e-12),dtype=self.dtype)
        
        if (nu == 1.5):
            poly = 1 +  tf.cast(tf.sqrt(3.0),dtype=self.dtype) * dist
        elif (nu == 2.5):
            poly = 1 +  tf.cast(tf.sqrt(5.0),dtype=self.dtype) * dist + dist_sq * (5.0 / 3.0)
        else:
            raise ValueError('Invalid nu (only 1.5 and 2.5 supported)')

        nu = tf.constant(nu, dtype=self.dtype)  
        K = poly * tf.exp(-tf.sqrt(2 * nu) * dist)
        
        if self.Verbose:
            print(f'Kernel Matrix:{K}')
            
        return K   

    
##############################################################################################
    def logLikelihood(self, X, y, alpha, noise, precision_matrix, kernel):
        """
        Function to compute the loglikelihood
        
        Args:
        -----
        alpha:  int
                Kernel's coVARIANCE amplitude parameter
                    
        noise:  
               noise VARIANCE parameter
               
        kernel: string
                kernel function to use, e.g. RBF or Matern.
                    
 
        Returns:
        --------
        log Likelihood:
                        logLikelihood of the data X,y given the kernel 
                        and it's hyperparmaters (amplitude, noise and precision matrix).  
                        
        #TODO: add to the log_lik differet dist==> poisson or student-T.
                
        """
        
        sizeX = int(X.shape[0])
        
        if kernel == 'rbf':
            K = self.kernel_rbf(X, X, precision_matrix)
        elif kernel == 'matern32':
            K = self.kernel_matern(X, X, precision_matrix, 1.5)
        elif kernel == 'matern52':
            K = self.kernel_matern(X, X, precision_matrix, 2.5)
                      
        KK = tf.abs(alpha) * K + (tf.abs(noise) + self.jitter) * tf.eye(sizeX, dtype=tf.float64)
        
        eignvals, eigenvectors = tf.linalg.eigh(KK)
        if self.Verbose:
            print('eignvals'+eignvals+'\neigenvectors',eigenvectors)
    
        # compute log-likelihood from kernel matrix
        chol = tf.linalg.cholesky(KK)
        Ky = tf.linalg.cholesky_solve(chol, y)
        log_lik = -0.5 * tf.matmul(tf.transpose(y), Ky)
        log_lik -= tf.reduce_sum(self.logguard(tf.linalg.diag_part(chol)))
        log_lik -= 0.5 * tf.cast(sizeX, tf.float64) * tf.cast(tf.math.log(2 * np.pi), tf.float64)
        
        
        if self.Verbose:
            print(f'log_likelihood:{log_lik[0][0]}')
        
        return log_lik[0][0] 
    
    
###############################################3################################################    
#     def train(self, lr = 0.005, niter = 5, optimizer = [tf.keras.optimizers.Adam, tf.optimizers.SGD, tf.optimizers.RMSprop][0],gradient_clipping=0, kernel=['rbf','matern32','matern52'][0], correlations = False, offset = False, monitor_period=None, monitor_gradient=None):
    def train(self, lr = 0.005, niter = 5, optimizer = [tf.compat.v1.train.AdamOptimizer, tf.compat.v1.train.GradientDescentOptimizer, tf.compat.v1.train.RMSPropOptimizer][0],gradient_clipping=0, kernel=['rbf','matern32','matern52'][0], correlations = False, offset = False, monitor_period=None, monitor_gradient=None):
        """
        Main function to train the kernel and its hyperparameters given data.
        The function minimizes the negative log likelihood using an optimizer (Adam ir GD)
        
        
        Args:
        -----
        lr:              learning rate (argument of optimizer)
        niter:           number of iterations to optimize over
        tol:             tolerance for which the iterations will break - i.e. log likelihood was converged.  
        optimizer:       function used to optimize the likelihood
        monitor_period:  how often (number of iterations) to print training statistics to montior training progress
        correlations:    Boolean. If True, train correlations of precision_matrix.
        offset:          Boolean. If True, add constant offset kernel and train.
        
        Returns:
        --------
        results:  List of ['logLikelihood', 'alpha', 'noise', 'lengthscales','gradients','precision_matrix'] 
                  for each optimization iteration

        """
        
        # define optimization variables 
        # prepare data
        X = tf.constant(self.X, dtype=self.dtype)
        y = tf.constant(self.y, dtype=self.dtype)
        if offset:
            offset=tf.Variable(np.zeros(1), dtype=tf.float64)
            y -= offset
            
        # kernel LOG parameters        
        log_lengthscales = tf.Variable(self.logguard(np.ones(self.dim)), dtype=self.dtype)
        log_alpha = tf.Variable(self.logguard(np.max(self.y) * np.ones(1)), dtype=self.dtype)    # amplitude coVariance parameter
        log_noise = tf.Variable(self.logguard(np.std(self.y)**2 * np.ones(1)), dtype=self.dtype) # noise VARIANCE param
                
        # kernel REAL parameters
        alpha = tf.exp(log_alpha)
        noise =  tf.exp(log_noise)
        lengthscales = tf.exp(log_lengthscales)
        precision_matrix = tf.linalg.diag(lengthscales**-2) # diagonal matrix of inverse square length scales
        if correlations:
            tiu_idx = np.triu_indices(self.dim, k = 1)
            # print('tiu_idx',tiu_idx)   
            corr = tf.Variable((0.1*(np.random.rand(np.shape(tiu_idx)[1]))), dtype=self.dtype)

            # Make indices and mask
            mask = np.zeros((self.dim,self.dim), dtype=bool)
            mask[tiu_idx] = True
            # print('mask',mask)

            idx = np.zeros((self.dim,self.dim), dtype=int)
            # print(np.shape(tiu_idx)[1])
            idx[tiu_idx] = np.arange(np.shape(tiu_idx)[1])
            # print('idx',idx)      

            # Make upper triangular matrix
            corr_matrix = tf.where(mask, tf.gather(corr, idx), tf.zeros((self.dim,self.dim), dtype=self.dtype))
            corr_matrix += tf.transpose(corr_matrix) # symmetrize
            precision_matrix += corr_matrix
        
        # calculate log likelihood
        loglik  = self.logLikelihood(X, y, alpha, noise, precision_matrix, kernel)

                  
        # define operations to maximize likelihood
        # NOTE - The optimizer runs on the log parameters for stability, but the likelihood is calculated on the real parameters
        opt = optimizer(learning_rate=lr) 
        
        # list of variables that optimizer can touch 
        train_vars = [log_alpha, log_noise, log_lengthscales] 
        if correlations:
            train_vars += [corr]
        if offset:
            train_vars += [offset]
              
        gradients = opt.compute_gradients(loss=-loglik, var_list=train_vars)
        if gradient_clipping:
            # apply gardient clipping
            print('<><><><><><> Gradinet clipper is applied <><><><><><><>')    
            clipped_gradients = [(tf.clip_by_value(grad, -1000, 1000), var) for grad, var in gradients]
                    # Could also try: 
#                     clipped_gradients = [(tf.clip_by_global_norm(grad,clip_norm=5.0), var) for grad, var in gradients]
#                     clipped_gradients = tf.clip_by_global_norm(gradients, 5.0)
            opt_operation = opt.apply_gradients(clipped_gradients)

        #             gradients, variables = zip(*opt.compute_gradients(loss=-lik))
        #             gradients, _ = tf.clip_by_global_norm(gradients, 10000.0)
        #             opt_operation = opt.apply_gradients(zip(gradients, variables))

        else:
            opt_operation = opt.apply_gradients(gradients) 
            
        # NOTE - we could use 
        # opt_operation = opt.minimize(loss=-loglik, var_list=train_vars) 
        # to compute the gradients and apply them in one function.

        
    
        # set up tf session
        fields = ['logLikelihood', 'alpha', 'noise', 'lengthscales','gradients','precision_matrix']
        if correlations:
            fields += ['corr']
        if offset:
            fields += ['offset']
        self.results = {f : [] for f in fields}
        
        with tf.compat.v1.Session() as sess:
            
            # initializes variables inside the session                
            sess.run(tf.compat.v1.global_variables_initializer())  
            
            with progressbar.ProgressBar(max_value=niter) as bar:
                
                if self.Verbose:
                    if correlations:
#                         print('corr_matrix',sess.run(corr_matrix))
                        print('corr',sess.run(corr))                   
#                     print('precision matrix ',sess.run(precision_matrix))
#                     print('check precision matrix is symetric: ',np.allclose(sess.run(precision_matrix), sess.run(tf.transpose(precision_matrix)), rtol=1e-05, atol=1e-08))
                    print('train_vars',sess.run(train_vars))
#                     print('K',sess.run(K))
                    print('loglik',sess.run(loglik))

                    #print trainable variables 
                    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))  
                    variables_names = [v.name for v in tf.trainable_variables()]
                    values = sess.run(variables_names)
                    for k, v in zip(variables_names, values):
                        print ("Variable: ", k)
                        print ("Shape: ", v.shape)
                        print (v)
#                     print('gradients',sess.run(gradients))
#                     print('opt_operation',sess.run(opt_operation))
                
                
                # run the optimization 
                for train_iter in range(niter):
                    bar.update(train_iter)
                    sess.run(opt_operation)  # one training step
                    
                    # collect current value of variables
                    iterlist =[loglik, alpha, noise, lengthscales, gradients, precision_matrix]  # collect current value of variables
                    if correlations:
                        iterlist += [corr]
                    if offset:
                        iterlist += [offset]
                        
                    itervar = sess.run(iterlist)  
                    
                    
                    for j,f in enumerate(fields):
                        self.results[f].append(itervar[j])

#                     # set up break condition
#                     if train_iter>5:         
#                         if (self.results['logLikelihood'][-1]-self.results['logLikelihood'][-2]) <= tol:     
#                             break 
                            
#                     # print to screen for monitor        
#                     if monitor_period is not None:
#                         if train_iter % monitor_period is 0:
#                             print('step '+str(train_iter)
#                                       +': log Likelihood = '+str(self.results['logLikelihood'][-1]) 
#                                       +'\tamp param = '+str(self.results['alpha'][-1])
#                                       +'\tnoise var param = '+str(np.sqrt(self.results['noise'][-1]))
#                                       + '\tlengthscales = '+str(self.results['lengthscales'][-1]) 
#                                       + '\tprecision matrix is symetirc' +str(np.allclose(sess.run(precision_matrix), sess.run(tf.transpose(precision_matrix)), rtol=1e-05, atol=1e-08)))
#                             if monitor_gradient is not None:
#                                 print('\tgradients='+str(self.results['gradients'][-1]) 
                                          
                                          

###############################################################################################
# [tf.optimizers.Adam, tf.optimizers.SGD, tf.optimizers.RMSprop]
    def evaluate(self, amp_param, noise_param_variance, lengthscales=None, precision_matrix=None, optimizer=tf.compat.v1.train.AdamOptimizer, kernel=['rbf','matern32','matern52'][0]):
        """
        Function to evaluate the loglikelihood and gradients 
        
        """
             
        # prepare data
        X = tf.constant(self.X, dtype=tf.float64)
        y = tf.constant(self.y, dtype=tf.float64)
         
        if precision_matrix is not None:
            lengthscales=np.diag(precision_matrix)
            precision_matrix = tf.Variable(precision_matrix)
            
        # kernel LOG parameters
        log_alpha = tf.Variable(np.log([amp_param]), dtype=tf.float64)
        log_noise = tf.Variable(np.log([noise_param_variance]), dtype=tf.float64)  # noise is the noise VARIANCE param
        log_lengthscales= tf.Variable(np.log(lengthscales), dtype=tf.float64)
        
       # kernel REAL parameters
        alpha = tf.exp(log_alpha)
        noise_var =  tf.exp(log_noise)
        lengthscales = tf.exp(log_lengthscales)
        
        if precision_matrix is None:
            precision_matrix =tf.diag(lengthscales**-2) # diagonal matrix of inverse square length scales
            
        # calculate log likelihood
        loglik = self.logLikelihood(X, y, alpha, noise_var, precision_matrix, kernel)
        opt = optimizer(learning_rate=0.001) 
        train_vars = [alpha, noise_var, lengthscales, precision_matrix] 
        gradients = opt.compute_gradients(loss=-loglik, var_list=train_vars)
                                          
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())    # initializes variables inside the session
            return sess.run([loglik, gradients])  
    
    ##############################################################################################
    def deriv2(self, amp_param, noise_param_variance, lengthscale_params, likelihood, negative_gradient, eps=1e-6):
        """
        Function to compute the parabolic taylor expansion parameters at a point.  
        This function is sensitive to the eps value, and it depends on the data. 
        
        
        Args:
        -----
        negative_gradient: from optimizer
        
        
        Returns:
        --------
        Parabola parameters
        
        
        """
        
        # prepare data
        X = tf.constant(self.X, dtype=tf.float64)
        y = tf.constant(self.y, dtype=tf.float64)
        
        # kernel parameters
        alpha = tf.Variable([amp_param], dtype=self.dtype)
        noise = tf.Variable([noise_param_variance], dtype=self.dtype)  # noise is the noise VARIANCE param
        nlengthscales = len(lengthscale_params)
        lengthscales = tf.Variable(lengthscale_params, dtype=self.dtype)
        precision_matrix =tf.diag(lengthscales**-2) # diagonal matrix of inverse square length scales
        
        # calculate log likelihood
        loglik = self.logLikelihood(X, y, alpha, noise, precision_matrix)
        
        lik_lengthscales = []
        
        # evaluate
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())    # initializes variables inside the session
            
            # variation wrt amp_param
            sess.run(alpha.assign_add(eps)) # add eps to the value to get some change
            itervar = sess.run([loglik, alpha, noise, lengthscales])  # collect current value of variables
            lik_alpha = itervar[0]
            sess.run(alpha.assign_add(-eps)) # remove eps from the value to return to start
            
            # variation wrt noise_param
            sess.run(noise.assign_add(eps)) # add eps to the value to get some change
            itervar = sess.run([loglik, alpha, noise, lengthscales])  # collect current value of variables
            lik_noise = itervar[0]
            sess.run(noise.assign_add(-eps)) # remove eps from the value to return to start
            
            # variation wrt lengthscales
            for i in range(nlengthscales):
                projection_vector = np.zeros(nlengthscales); projection_vector[i] = 1; eps_vector = eps*projection_vector
                sess.run(lengthscales.assign_add(eps_vector)) # add eps to the value to get some change
                itervar = sess.run([loglik, alpha, noise, lengthscales])  # collect current value of variables
                lik_lengthscales += [itervar[0]]
                sess.run(lengthscales.assign_add(-eps_vector)) # remove eps from the value to return to start
                  
        def calcmygrad(x0, x1, y0, y1, g0): # g0 is the derivative dy/dx at x0
            denom = (x0 - x1)**2
            a0 = (g0*x0*(x0 - x1)*x1 - 2*x0*x1*y0 + x1**2*y0 + x0**2*y1)/denom
            a1 = (g0*(-x0**2 + x1**2) + 2*x0*(y0 - y1))/denom
            a2 = (g0*(x0 - x1) - y0 + y1)/denom
            return np.array([a0,a1,a2,-a1/(2*a2),-2*a2]) #-a1/(2*a2) is the peak location ,-2*a2 is the inverse of the width**2
                                          
                                          
        # calc and collect the parabola params
        returned_parabola_params = [calcmygrad(amp_param, amp_param+eps, likelihood, lik_alpha, -negative_gradient[0][0])] # amp
        returned_parabola_params += [calcmygrad(noise_param_variance, noise_param_variance+eps, likelihood, lik_noise, -negative_gradient[1][0])] # noise
        length_scale_parabola_params = [] # collect an array of arrays for each length scale
        
        for i in range(nlengthscales):
            # one list of parabola params for each length scale
            length_scale_parabola_params += [calcmygrad(lengthscale_params[i], lengthscale_params[i]+eps, likelihood, lik_lengthscales[i], -negative_gradient[2][0][i])] 
        returned_parabola_params += [length_scale_parabola_params]
       
        return returned_parabola_params
        
                                                             
    def parabola_approx(self, amp_param, noise_param_variance, lengthscale_params, likelihood, negative_gradient, kernel=['rbf','matern32','matern52'][0]):
        """
        Function to compute the parabolic taylor expansion parameters at a point.  
        This function is sensitive to the eps value, and it depends on the data. 
        
        
        Args:
        -----
        negative_gradient: from optimizer
        
        
        Returns:
        --------
        Parabola parameters
        """
                                          
        # prepare data
        X = tf.constant(self.X, dtype=tf.float64)
        y = tf.constant(self.y, dtype=tf.float64)
        
        # kernel parameters
        alpha = tf.Variable([amp_param], dtype=tf.float64)
        noise = tf.Variable([noise_param_variance], dtype=tf.float64)  # noise is the noise VARIANCE param
        nlengthscales=len(lengthscale_params)
        lengthscales = tf.Variable(lengthscale_params, dtype=tf.float64)
        precision_matrix =tf.compat.v1.diag(lengthscales**-2) # diagonal matrix of inverse square length scales
        
        # calculate log likelihood
        lik = self.logLikelihood(X, y, alpha, noise, precision_matrix,kernel)
        
        hessian = tf.hessians(lik, lengthscales)
        hessian_amp = tf.hessians(lik, alpha)
        hessian_noise = tf.hessians(lik, noise)
        
        def calcmygrad(x0, y0, g0, h0): # g0 is the derivative dy/dx at x0
            a0 = -g0*x0+(h0*x0**2)/2+y0
            a1 = g0-h0*x0
            a2 = h0/2
            return np.array([a0,a1,a2,-a1/(2*a2),np.sqrt(1/(-2*a2))]) #-a1/(2*a2) is the peak location ,sqrt(1/(-2*a2)) is the width
       
        # evaluate
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())    # initializes variables inside the session
            hess_matrix=sess.run(hessian)
            hess_amp=sess.run(hessian_amp)
            hess_noise=sess.run(hessian_noise)
       
       # calc and collect the parabola params
        returned_parabola_params =[]
        returned_parabola_params += [calcmygrad(amp_param, likelihood, -negative_gradient[0][0][0],hess_amp[0][0][0])] # amp
        returned_parabola_params += [calcmygrad(noise_param_variance, likelihood, -negative_gradient[1][0][0],hess_noise[0][0][0])] # noise
        
#         length_scale_parabola_params = [hess_matrix]  # collect an array of arrays for each length scale
        length_scale_parabola_params = []  # collect an array of arrays for each length scale
        for i in range(nlengthscales):
            length_scale_parabola_params += [calcmygrad(lengthscale_params[i], likelihood, -negative_gradient[2][0][i],np.diag(hess_matrix[0])[i])]
        
        returned_parabola_params += [length_scale_parabola_params]
        
        return returned_parabola_params