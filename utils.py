import matplotlib.pyplot as plt
import numpy as np
import pickle  
import warnings
import time
warnings.filterwarnings("ignore")
font = {'size'   : 14}
plt.rc('font', **font)


def GP_plot_results(tfgp,yin):
    ll=tfgp.results['logLikelihood']
    likelihood=np.exp(ll)
    amp_param=tfgp.results['alpha']
    noise_param_variance=(tfgp.results['noise'])
    length_scale_param=tfgp.results['lengthscales']
    try:
        offset_param=tfgp.results['offset']
        corr_param=tfgp.results['corr']
    except:
        offset_param=None
        corr_param=None
    gradients=tfgp.results['gradients']
    ls_grad=np.array([a[2][0] for a in gradients])
    precision_matrix=tfgp.results['precision_matrix']
    Parabola_vals=tfgp.parabola_approx(amp_param[-1][0], noise_param_variance[-1][0], length_scale_param[-1], ll[-1], gradients[-1],kernel='matern32')

    plt.figure(figsize=(20,8));
    plt.subplot(241); plt.plot(np.array(ll)/len(yin)); plt.xlabel('step'); plt.ylabel('log likelihood / npts')
    # plt.subplot(231); plt.plot(ll); plt.xlabel('step'); plt.ylabel('log likelihood')
    plt.subplot(242); plt.plot(amp_param); plt.xlabel('step'); plt.ylabel('amp param');
    plt.subplot(243); plt.plot(noise_param_variance); plt.xlabel('step'); plt.ylabel('noise param variance');
    plt.subplot(244); plt.plot(length_scale_param); plt.xlabel('step'); plt.ylabel('lengthscales');
    plt.subplot(245); plt.plot(ls_grad); plt.xlabel('step'); plt.ylabel('length scale gradient');
    
    if offset_param is not None:
        try:
            plt.subplot(246); plt.plot(offset_param); plt.xlabel('step'); plt.ylabel('offset param')
        except:
            pass
    if corr_param is not None:
        try:
            plt.subplot(247);  plt.plot(corr_param); plt.xlabel('step'); plt.ylabel('corr param')
        except:
            pass
    plt.tight_layout()
    plt.show()
    
    results={}
    print('logLiklihod',ll[-1],'\n amp',amp_param[-1][0], '\n noise' ,noise_param_variance[-1][0],'\n ls', length_scale_param[-1])
    try:
        print('\ncorr_param',corr_param[-1])
        results['corr_param'] = corr_param[-1]       
    except:
        results['corr_param'] = None
    try:
        print('\noffset',offset_param[-1][0])
        results['offset_param'] = offset_param[-1][0] 
    except:
        results['offset_param'] = None
    results['ll'] = -ll[-1] 
    results['precision_matrix'] = precision_matrix[-1] 
    results['amp_param'] = amp_param[-1][0] 
    results['noise_param_variance'] = noise_param_variance[-1][0] 
    results['length_scale_param'] = length_scale_param[-1] 
    return results


def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins) #bincount - Count number of occurrences of each value in uniq_keys

 
def save_obj(str_name,var):
    filehandler = open('Results/'+str_name+'.pkl', 'wb')
    pickle.dump(var, filehandler)
    filehandler.close()
    
def func(x,Parabola_vals):
    par_tot=[]
    a0=Parabola_vals[0]
    a1=Parabola_vals[1]
    a2=Parabola_vals[2]
    xp=-Parabola_vals[3]
#     par_tot = a0+a1*(x)+a2*(x)**2
    par_tot = a0+a1*(x-xp)+a2*(x-xp)**2
    #             par_tot += [a0+a1*(x-xp)+a2*(x-xp)**2] #[a0+a1*(x)+a2*(x)**2]
    return par_tot

def width_calc(keyplt):
    sig_p=[]
    sigi=float(Parabola_vals_tot.loc[keyplt]['scan1'][4])
    sig_p=np.append(sig_p, sigi)
    return sig_p

def Xp_calc(keyplt):
    Xp=[]
    tmp=float(Parabola_vals_tot.loc[keyplt]['scan1'][3])
    Xp=np.append(Xp, tmp)
    return Xp

def gauss_fit(x, A, sig, mu):
    return A*np.exp(-0.5*(x-mu)**2/sig**2)

def gauss_fit_offset(x, A, sig, mu, offset):
    return A*np.exp(-0.5*(x-mu)**2/sig**2)+offset

