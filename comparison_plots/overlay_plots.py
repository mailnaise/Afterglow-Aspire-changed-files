import json
import numpy as np
import corner
import matplotlib.pyplot as plt
import os

def load_bilby_posterior(path, keys=None):
    path = os.path.expanduser(path)
    with open(path) as f:
        data = json.load(f)
    content = data['posterior']['content']
    keys = keys or data['search_parameter_keys']  # use fitted params by default
    samples = np.array([content[k] for k in keys]).T  # shape (n_samples, n_params)
    return samples, keys

samples3, keys3 = load_bilby_posterior('../Results/AutoDoc/2026_07_03/schecter_luminosity/afterglow_GW170817_GJ/12_50_55_XrayNessai/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')
#samples2, keys2 = load_bilby_posterior('../Results/AutoDoc/2026_07_03/schecter_luminosity/afterglow_GW170817_GJ/12_51_28_XrayAspire/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')

#samples2, keys2 = load_bilby_posterior('../Results/AutoDoc/2026_07_03/schecter_luminosity/GW170817_GJ/10_01_01_GWGRBNessai/temp/GJ/1/trace/GW170817_GJ_result.json')
#samples2, keys2 = load_bilby_posterior('../Results/AutoDoc/2026_07_14/schecter_luminosity/afterglow_GW170817_GJ/14_39_52_XrayAspireFixed/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')

#samples1, keys1 = load_bilby_posterior('../Results/AutoDoc/2026_07_06/schecter_luminosity/afterglow_GW170817_GJ/10_57_34_DynestyXray/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')

#samples1, keys1 = load_bilby_posterior('../Results/AutoDoc/2026_07_06/schecter_luminosity/GW170817_GJ/10_57_26_DynestyGWGRB/temp/GJ/1/trace/GW170817_GJ_result.json')
#samples1, keys1 = load_bilby_posterior('~/MSc/Afterglow_Aspire/Results/AutoDoc/2026_07_03/schecter_luminosity/GW170817_GJ/12_51_24_GWGRBAspire/temp/GJ/1/trace/GW170817_GJ_result.json')

#Aspire with different init comparison
samples2, keys2 = load_bilby_posterior('~/zips/14_23_54_xrayaspirefastes_wdynt/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')
#samples2, keys2 = load_bilby_posterior('~/zips/13_03_02_xrayfast_nessaiinit/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')

samples1, keys1 = load_bilby_posterior('~/MSc/Afterglow_Aspire/Results/AutoDoc/2026_07_23/schecter_luminosity/afterglow_GW170817_GJ/14_24_09_xraydyn_w16cores/temp/GJ/1/trace/afterglow_GW170817_GJ_result.json')




fig = corner.corner(samples1, labels=keys1, color='blue', show_titles=True, hist_kwargs=dict(density=True))
corner.corner(samples2, fig=fig, color='red', hist_kwargs=dict(density=True))
corner.corner(samples3, fig=fig, color='green', hist_kwargs=dict(density=True))

plt.legend(
    handles=[
        plt.Line2D([], [], color='blue', label='Dynesty'),
        plt.Line2D([], [], color='red', label='Aspire (Dynesty Flow)'),
        plt.Line2D([], [], color='green', label='Nessai'),
    ],
    loc='upper right',
    bbox_to_anchor=(1, len(keys1))
)

#fig.savefig('xray_nessai_vs_aspire.png', dpi=300, bbox_inches='tight')
#fig.savefig('GWGRB_nessai_vs_aspire.png', dpi=300, bbox_inches='tight')
fig.savefig('XRAY_allthreesamplers.png', dpi=300, bbox_inches='tight')