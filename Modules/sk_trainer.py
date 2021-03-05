import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel


class sk_trainer:
        '''
        This clasee uses sklearn package to train the GP kernel's hyperparameters
        '''
    
        def __init__(self):
            self.results={}


        def SK_kernel(self, kern = ['rbf','matern32','matern52'][0],length=1):  
            if kern == 'matern32':
                return Ck(10.0)*Matern(length_scale=np.array([1.0]*length), length_scale_bounds=(0.0000001,20),nu=1.5)
            elif kern == 'matern52':
                return Ck(10.0)*Matern(length_scale=np.array([1.0]*length), length_scale_bounds=(0.0000001,20),nu=2.5)
            elif kern == 'rbf':
                return Ck(10.0)*RBF(length_scale=np.array([1.0]*length), length_scale_bounds=(0.0000001,20))


        def sk_train(self,X_train, y_train, kernels= ['rbf','matern32','matern52'],offset=False,verboseQ=False,saveQ=True):
            for kern in (kernels):
                if verboseQ: print('********* \n',kern)
                sk_kernel = self.SK_kernel(kern,X_train.shape[1]) 
                sk_kernel += WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-8, 10))  # noise 

                if offset: #add bias kernel
                    sk_kernel += Ck(1) 

                t0=time.time()
                gpr = GaussianProcessRegressor(kernel = sk_kernel, n_restarts_optimizer = 5)
                gpr.fit(X_train, y_train)
                self.sk_t = time.time() - t0

                if verboseQ:
                    print ('took ',self.sk_t  ,' seconds')
                    print("Inital kernel: %s" % gpr.kernel)
                    print("Learned kernel: %s" % gpr.kernel_)
                    print("Log-marginal-likelihood: %.3f" % gpr.log_marginal_likelihood(gpr.kernel_.theta))

                if saveQ:
                    with open('SK_gpr_'+str(kern)+'.pickle', 'wb') as handle:
                        pickle.dump(gpr, handle, protocol=pickle.HIGHEST_PROTOCOL)

            
            self.results['ll'] =  gpr.log_marginal_likelihood(gpr.kernel_.theta)
            try:
                self.results['amp_param'] = gpr.kernel_.get_params()['k1__k1__k1__constant_value']
                self.results['noise_param_variance'] = gpr.kernel_.get_params()['k1__k2__noise_level']
                self.results['length_scale_param'] = gpr.kernel_.get_params()['k1__k1__k2__length_scale'].tolist()
                self.results['offset_param'] = gpr.kernel_.get_params()['k2__constant_value']  
            except:
                self.results['length_scale_param'] = gpr.kernel_.get_params()['k1__k2__length_scale'].tolist()
                self.results['noise_param_variance'] = gpr.kernel_.get_params()['k2__noise_level']
                self.results['amp_param'] = gpr.kernel_.get_params()['k1__k1__constant_value']   
                self.results['offset_param'] = None

                