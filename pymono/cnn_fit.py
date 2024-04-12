import numpy as np 
from scipy.optimize import curve_fit
from collections import namedtuple
import matplotlib.pyplot as plt

def bin_centers(bins):
    return np.array([(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)])
def weighted_error(sigma1, w1, sigma2, w2):
    return sigma1 * w1 + sigma2 * w2


def fit_dgaussian(xdata, 
                  ydata, 
                  pars=[10000., 0.5, 2.0, 10000., 0.5, 2.0],
                  lower_bounds = [0, -1.0, 0.0, 0.0, -1.0, 0.0],
                  upper_bounds = [100000, 1.0, 15.0, 100000.0, 1.0, 15.0]
                 ):

    def gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
        g1 = a1*(np.exp(-(x-mu1)**2/(2*sigma1**2)))
        g2 = a2*(np.exp(-(x-mu2)**2/(2*sigma2**2)))
        return g1 + g2
    
    pars, cov = curve_fit(gaussian, xdata, ydata, p0=pars, 
                          bounds=(lower_bounds, upper_bounds))
    err       = np.sqrt(np.diag(cov))
    fit_y     = gaussian(xdata, *pars) 
    return pars, err, fit_y 


def fit_coord(tdeltas, bins=100):
    def bin_data(data, bins):
        hdz, binsz = np.histogram(data, bins=bins)
        xdata = bin_centers(binsz)
        ydata = hdz
        return xdata, ydata

    def fitcoord2g(data, bins):
        xdata, ydata = bin_data(data, bins)
        pars, err, yfit = fit_dgaussian(xdata, ydata)
        n1 = pars[0]/(pars[0]+ pars[3])
        n2 = pars[3]/(pars[0]+ pars[3])
        we = weighted_error(pars[2], n1, pars[5], n2)
        print(f"2g fit: mu1 = {pars[1]:.2f}, sigma = {pars[2]:.2f}, n1  ={n1:.2f}")
        print(f"2g fit: mu2 = {pars[4]:.2f}, sigma = {pars[5]:.2f}, n1  ={n2:.2f}")
        print(f"weighted error: = {we:.2f}")
    
        fit2g = namedtuple('fit2g',
               'xdata, ydata, yfit, norms, mus, sigmas')
        return fit2g(xdata, ydata, yfit, 
                    (pars[0], pars[3]), (pars[1], pars[4]), (pars[2], pars[5]))
        
    f2gz = fitcoord2g(tdeltas.delta_z_NN, bins=bins)
    f2gx = fitcoord2g(tdeltas.delta_x_NN, bins=bins)
    f2gy = fitcoord2g(tdeltas.delta_y_NN, bins=bins)
    
    return f2gz, f2gx, f2gy


def fit_coord(X, Y, Z,  bins=100):
    def bin_data(data, bins):
        hdz, binsz = np.histogram(data, bins=bins)
        xdata = bin_centers(binsz)
        ydata = hdz
        return xdata, ydata

    def fitcoord2g(data, bins):
        xdata, ydata = bin_data(data, bins)
        pars, err, yfit = fit_dgaussian(xdata, ydata)
        n1 = pars[0]/(pars[0]+ pars[3])
        n2 = pars[3]/(pars[0]+ pars[3])
        we = weighted_error(pars[2], n1, pars[5], n2)
        print(f"2g fit: mu1 = {pars[1]:.2f}, sigma = {pars[2]:.2f}, n1  ={n1:.2f}")
        print(f"2g fit: mu2 = {pars[4]:.2f}, sigma = {pars[5]:.2f}, n1  ={n2:.2f}")
        print(f"weighted error: = {we:.2f}")
    
        fit2g = namedtuple('fit2g',
               'xdata, ydata, yfit, norms, mus, sigmas')
        return fit2g(xdata, ydata, yfit, 
                    (pars[0], pars[3]), (pars[1], pars[4]), (pars[2], pars[5]))
        
    f2gz = fitcoord2g(Z, bins=bins)
    f2gx = fitcoord2g(X, bins=bins)
    f2gy = fitcoord2g(Y, bins=bins)
    
    return f2gz, f2gx, f2gy


def plotfxyz(f2gx, f2gy, f2gz, figsize=(8, 6)):
    def n1n2(norms):
        n1 = norms[0]/(norms[0]+ norms[1])
        n2 = norms[1]/(norms[0]+ norms[1])
        return n1, n2
        
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    flat_axes = axes.ravel()
    ax0, ax1, ax2 = flat_axes[0], flat_axes[1], flat_axes[2]

    n1, n2 = n1n2(f2gx.norms)
    ax0.errorbar(f2gx.xdata, f2gx.ydata,yerr=np.sqrt(f2gx.ydata), fmt='o', markersize=2)
    ax0.plot(f2gx.xdata, f2gx.yfit, 'red', linewidth=1.0,
                label=f"$\sigma_1$(i={n1:.2f}) = {f2gx.sigmas[0]:.2f}\n $\sigma_2$(i={n2:.2f}) = {f2gx.sigmas[1]:.2f}")
    ax0.legend()
    ax0.set_xlabel("NN (xtrue - xpredicted)",fontsize=14)
    ax0.set_ylabel("Counts/bin",fontsize=14)
    
    n1, n2 = n1n2(f2gy.norms)
    ax1.errorbar(f2gy.xdata, f2gy.ydata,yerr=np.sqrt(f2gy.ydata), fmt='o', markersize=2)
    ax1.plot(f2gy.xdata, f2gy.yfit, 'red', linewidth=1.0,
                label=f"$\sigma_1$(i={n1:.2f}) = {f2gy.sigmas[0]:.2f}\n $\sigma_2$(i={n2:.2f}) = {f2gy.sigmas[1]:.2f}")
    ax1.legend()
    ax1.set_xlabel("NN (ytrue - ypredicted)",fontsize=14)
    ax1.set_ylabel("Counts/bin",fontsize=14)

    n1, n2 = n1n2(f2gz.norms)
    ax2.errorbar(f2gz.xdata, f2gz.ydata,yerr=np.sqrt(f2gz.ydata), fmt='o', markersize=2)
    ax2.plot(f2gz.xdata, f2gz.yfit, 'red', linewidth=1.0,
                label=f"$\sigma_1$(i={n1:.2f}) = {f2gz.sigmas[0]:.2f}\n $\sigma_2$(i={n2:.2f}) = {f2gz.sigmas[1]:.2f}")
    ax2.legend()
    ax2.set_xlabel("NN (ztrue - zpredicted)",fontsize=14)
    ax2.set_ylabel("Counts/bin",fontsize=14)

    fig.tight_layout()
    plt.show()