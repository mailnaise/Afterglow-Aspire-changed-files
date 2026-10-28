"""
Model Comparison
=============================

This file initialises the sampling of the different models.

Copyright 2018 Fergus Hayes

"""

import matplotlib
matplotlib.use("agg")

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from aspire import Aspire
from aspire_bilby.utils import samples_from_bilby_result, get_aspire_functions
from matplotlib.ticker import FormatStrFormatter
from models.models import JS_model_interp_nessai
import argparse, os, time

np.random.seed(int(time.time()))

def main(path, beta, structure, burn=300000, draws=100000, progressbar=False, interp=False, cores=None, Lfixed=False, default=True, sampler='aspire'):

    ######### Sample model #########

    Y = []
    F_sig = []
    y_afg = []
    f_afg_sig = [] 
    d = []
    cosi = []
    Nz = False
 
    all_file = np.sort(os.listdir(path))
    for datafile in all_file:
        if datafile[-8:]=='Flux.npy':
            Y.append(np.load(path+datafile).squeeze())
            F_sig.append(np.load(path+datafile[:-8]+'Flux_uncertainty.npy').squeeze())
            d_, cosi_ = np.load(path+datafile[:-8]+'dcosi_samples.npy')
            d.append(d_)
            cosi.append(cosi_)
        elif datafile=='rates.npy':
            Nz = np.load(path+datafile)
        elif datafile[-8:]=='xray.npy':
            y_afg.append(np.load(path+datafile)[:9].squeeze()) 
            f_afg_sig.append(np.load(path+datafile[:-8]+'xray_uncertainty.npy')[:9].squeeze())

   # for datafile in all_file:
   #     if 'Flux' in datafile:
   #         if 'xray' in datafile:
   #             if 'uncertainty' in datafile:
   #                 #f_afg_sig.append(np.load(path+datafile)[7:].squeeze())
   #                 f_afg_sig.append(np.load(path+datafile)[:9].squeeze())
   #             #y_afg, f_afg_sig = y_afg[7:], f_afg_sig[7:]
   #             else:
   #                 #y_afg.append(np.load(path+datafile)[7:].squeeze())
   #                 y_afg.append(np.load(path+datafile)[:9].squeeze())
   #             #print(y_afg, f_afg_sig)
   #         else:
   #             if 'uncertainty' in datafile:
   #                 F_sig.append(np.load(path+datafile).squeeze())
   #             else:
   #                 Y.append(np.load(path+datafile).squeeze())
   #     elif 'dcosi' in datafile:
   #         print(datafile)
   #         d_, cosi_ = np.load(path+datafile)
   #         d.append(d_)
   #         cosi.append(cosi_)
   #     elif datafile=='rates.npy':
   #         Nz = np.load(path+datafile)
    
    d, thetav = np.array(d), np.arccos(cosi) 
    if len(Y)==0:
        Y, F_sig = False, False
        #Y, F_sig, y_afg, f_afg_sig, d, thetav = False, False, False, False, False, False

    else:
        Y, F_sig, = np.array(Y), np.array(F_sig)
    
    if len(y_afg)==0:
        y_afg, f_afg_sig = False, False
    else:
        y_afg, f_afg_sig = np.array(y_afg), abs(np.array(f_afg_sig))
        print('y_afg.shape: ', y_afg.shape)
        #y_afg, f_afg_sig = np.array(y_afg)[3:], abs(np.array(f_afg_sig)[3:]) #Run with short LC
 
    print('d.shape: ', d.shape)
    print('Y.shape: ', Y.shape)
    print('file reading: ', 'y_afg: ', y_afg, 'f_afg_sig: ', f_afg_sig)
    #print('d: ', d)
    #print('thetav: ', thetav)
    
    if isinstance(cores,str):
        cores = int(cores)

    if cores==None:
        n_pool=None
    elif cores==1:
        n_pool=None
    elif cores>1:
        n_pool=cores-1

    if not structure:
        structure = str(path.split('/')[-3].split('_')[0])

    path = path+'temp/'+str(structure)+'/'+str(int(np.round(beta)))+'/'

    # load or create random seed
    if os.path.exists(path+'random_seed.npy'):
        random_seed = int(np.load(path+'random_seed.npy'))
    else:
        random_seed = np.random.randint(1.e8)
        np.save(path+'random_seed',random_seed)

    # Run sampler on given jet structure models. The models are specified in "./models/models.py"
    result = JS_model_interp_nessai(structure, thetav, Y, d, F_sig, y_afg, f_afg_sig, beta=beta, outdir=path+'trace/', label=path.split('/')[-6], Nz_obs=Nz, burn=burn, draws=draws, progressbar=False, random_seed=random_seed, interp=interp, sampler=sampler, cores=cores, n_pool=n_pool, Lfixed=Lfixed, default=default)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='', description="Runs MCMC given structure model.")
    parser.add_argument('--path', help="Path to run.")
    parser.add_argument('--sampler', help="Which sampler to use (nessai, dynesty, aspire).", default='aspire')
    parser.add_argument('--beta', help="Temperature of sampling.", default='1')
    parser.add_argument('--struct', help="Structure to analyse.", default=False)
    parser.add_argument('--burn', help="Number of samples to discard.", default=500000)
    parser.add_argument('--draws', help="Number of samples to keep.", default=100000)
    parser.add_argument('--progressbar', help="See progressbar.", default=False)
    parser.add_argument('--interp', help="Interpolation.", action='store_true', default=False) 
    parser.add_argument('--Lfixed', help="Fix luminosity function.", action='store_true', default=False)
    parser.add_argument('--cores', help="Number of cores to use.", default=None)
    parser.add_argument('--ndefault', help="See progressbar.", action='store_false', default=True)

    opt = parser.parse_args()

    main(opt.path, float(opt.beta), opt.struct, int(opt.burn), int(opt.draws), bool(opt.progressbar), interp=opt.interp, cores=opt.cores,Lfixed=opt.Lfixed, default=opt.ndefault,sampler=opt.sampler)
