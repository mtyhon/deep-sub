import numpy as np
import matplotlib.pyplot as plt


def plot_10d(par_pi, par_mus, par_sigmas, output_grid, output_median, output_conf_interval, output_pdf):

    fig = plt.figure(figsize=(25, 10))
    par_name = ['$\\tau$', '$M$', '$Y_0$', '$Z_0$','$\\alpha_{\\mathrm{MLT}}$', '$\\alpha_{\\mathrm{over}}$', '$D$',
            '$\\alpha_{\\mathrm{under}}$', '$R$', '$L$']
    par_units = ['Gyr', '$M_{\\odot}$', '', '','','','','','$R_{\\odot}$', '$L_{\\odot}$' ]        

    for i in range(len(par_mus)):
        exec("ax%d = fig.add_subplot(2,5,%d)" %(i+1, i+1))
        exec("ax%d.tick_params(width=5)" %(i+1))
        for axis in ['top','bottom','left','right']:
            exec("ax%d.spines[axis].set_linewidth(2)" %(i+1))
 

        exec("ax%d.plot(output_grid[i], output_pdf[i], c='k')" %(i+1))
        exec("ax%d.text(x=0.75, y=0.9, s=par_name[i], transform = ax%d.transAxes, fontsize=27)" %(i+1, i+1))


        if i == 0:
            exec("ax%d.set_xlim([max(0,output_median[i] - 7*np.median(par_sigmas[i])), output_median[i] + 7*np.median(par_sigmas[i])])" %(i+1))
        elif i == 4:
            exec("ax%d.set_xlim([max(0,output_median[i] - 8*np.median(par_sigmas[i])), output_median[i] + 8*np.median(par_sigmas[i])])" %(i+1))        

        elif i == 2:
            exec("ax%d.set_xlim([max(0,output_median[i] - 5*np.median(par_sigmas[i])), output_median[i] + 5*np.median(par_sigmas[i])])" %(i+1))        
        
        else:
            exec("ax%d.set_xlim([max(0,output_median[i] - 5*np.median(par_sigmas[i])), output_median[i] + 5*np.median(par_sigmas[i])])" %(i+1))
            if i == 3:
               pass
        
        exec("ax%d.set_xlabel(par_units[i], fontsize=27)" %(i+1))

        exec("ax%d.set_yticklabels([])" %(i+1))
        exec("ax%d.tick_params(axis='x', labelsize=30)" %(i+1))

        if i in [5,6,7]:
            exec("ax%d.set_xscale('log')" %(i+1))
            if i != 6:
                exec("ax%d.set_xlim([1e-4, 1])" %(i+1))
            else:
                exec("ax%d.set_xlim([1e-4, 3])" %(i+1))
            exec("ax%d.vlines(x=output_conf_interval[i], ymin=[-0.05,-0.05],\
        ymax= [1e6, 1e6],\
          linestyles='dotted', lw=3)" %(i+1))
            exec("ax%d.set_ylim([-0.05, np.max(output_pdf[i]) + 10])" %(i+1))
            exec("ax%d.set_xticks([1e-4, 1e-2, 1])" %(i+1))

            exec("ax%d.vlines(x=output_median[i], linestyles='dashed', ymin=-0.05,\
    ymax = 1e6)" %(i+1))
        else:
            exec("ax%d.vlines(x=output_conf_interval[i], ymin=[-0.05,-0.05],\
    ymax=[output_pdf[i][np.argmin(np.abs(output_grid[i]-output_conf_interval[i][0]))],\
    output_pdf[i][np.argmin(np.abs(output_grid[i]-output_conf_interval[i][1]))]],\
          linestyles='dotted', lw=3)" %(i+1))
            exec("ax%d.set_ylim([-0.05, 1.1*np.max(output_pdf[i])])" %(i+1))
            exec("ax%d.vlines(x=output_median[i], linestyles='dashed', ymin=-0.05,\
    ymax = output_pdf[i][np.argmin(np.abs(output_grid[i]-output_median[i]))], lw=3)" %(i+1))

    
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    plt.grid(False)
    plt.ylabel("Density", fontsize=30)
    plt.tight_layout(w_pad=0.25)
    plt.savefig('results.png')
    plt.close()


