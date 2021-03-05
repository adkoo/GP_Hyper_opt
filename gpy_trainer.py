import GPy
import time
import numpy as np

class gpy_trainer:
        '''
        This clasee uses GPy package to train the GP kernel's hyperparameters
        '''
    
        def __init__(self):
            self.results={}

        def gpy_kernel(self,kern = ['rbf','matern32','matern52'][0],length=1):  
            if kern == 'matern32':
                return GPy.kern.Matern32(length, ARD = 1)
            elif kern == 'matern52':
                return GPy.kern.Matern52(length, ARD = 1)
            elif kern == 'rbf':
                return GPy.kern.RBF(length, ARD = 1)

        def gpy_train(self,X_train, y_train, kernels= ['rbf','matern32','matern52'],max_iters=1000,offset=False,verboseQ=False,saveQ=True):
            for kern in (kernels):
                if verboseQ: print('********* \n',kern)
                gp_kernel = self.gpy_kernel(kern, X_train.shape[1]) 
                gp_kernel += GPy.kern.White(X_train.shape[1])

                if offset: #add bias kernel
                    gp_kernel += GPy.kern.Bias(1)

                t0=time.time()
                gpr = GPy.models.GPRegression(X_train, y_train, gp_kernel)
                gpr.optimize(optimizer='lbfgs', max_iters=max_iters)
                gpr.optimize_restarts(num_restarts = 5)
                self.gpy_t = time.time()-t0


                if verboseQ:
                    print ('took ',self.gpy_t ,' seconds')
                    print("Log-marginal-likelihood: %.3f" %  -gpr.objective_function())

                if saveQ:
                    with open('GPY_gpr_'+str(kern)+'.pickle', 'wb') as handle:
                        pickle.dump(gpr, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.results['ll'] = -gpr.objective_function()
            self.results['amp_param'] = np.array(gpr.to_dict()['kernel']['parts'][0]['variance'])[0]
            self.results['noise_param_variance'] = gpr.to_dict()['kernel']['parts'][1]['variance'][0]
            self.results['length_scale_param'] =  np.array(gpr.to_dict()['kernel']['parts'][0]['lengthscale'])
            try:
                self.results['offset_param'] = np.array(gpr.to_dict()['kernel']['parts'][2]['variance'])[0]
            except:
                self.results['offset_param'] = None
