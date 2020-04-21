from model import *
from utils.load_input import load_mode_data
from utils.image_array import generate_image
from utils.plot import *
from sklearn.linear_model import LinearRegression

import os, torch, argparse
import scipy as sp

parser = argparse.ArgumentParser()

parser.add_argument("--star_id", type=int, required=True)
parser.add_argument("--mode_filename", type=str, required=True)
parser.add_argument("--teff", type=int, required=True)
parser.add_argument("--teff_sig", type=int, required=True)
parser.add_argument("--numax", type=float, required=True)
parser.add_argument("--numax_sig", type=float, required=True)
parser.add_argument("--fe_h", type=float, required=True)
parser.add_argument("--fe_h_sig", type=float, required=True)
parser.add_argument("--output_10d",type=bool, default=False)
parser.add_argument("--num_samples", type=int, default=5000)

config = parser.parse_args()

def infer(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    package_dir = os.path.dirname(os.path.abspath(__file__))
    saved_model_dict = package_dir +'/trained_models/publish_model.torchmodel'
    trained_model = Echelle_Conv2D_Array(hidden_size=512, kernel_size=5, num_gaussians=16).to(device)
    trained_model.load_state_dict(torch.load(saved_model_dict))
    trained_model.eval()

    mode_star_id, (radial_modes, dipole_modes, quad_modes) = load_mode_data(config.mode_filename)
    assert mode_star_id == config.star_id

    empirical_dnu = 0.262*(config.numax**0.772)
    radial_closest = np.sort(radial_modes[np.argsort(np.abs(radial_modes - config.numax))[:6]])
    diff  =  np.round(np.diff(radial_closest)/empirical_dnu)
    insert = np.insert(diff, 0, 0)
    order_closest = np.cumsum(insert)
    WLS = LinearRegression()
    fwhm = 2*np.sqrt(2*np.log(2))
    weights = sp.stats.norm.pdf(radial_closest, config.numax, 0.25*config.numax/fwhm)
    WLS.fit(order_closest.reshape(-1, 1), radial_closest.reshape(-1, 1), sample_weight=weights)

    delta_nu = WLS.coef_.squeeze()
    epsilon = WLS.intercept_/delta_nu
    if epsilon >= 1:
        epsilon = epsilon % 1

    combined_freqs = np.concatenate([radial_modes, dipole_modes]) 
    combined_freqs = np.concatenate([combined_freqs, quad_modes])
    combined_deg = np.concatenate([np.zeros(len(radial_modes)), np.ones_like(dipole_modes)])
    combined_deg = np.concatenate([combined_deg, np.ones_like(quad_modes) + 1])

    imgs = generate_image(combined_freqs, combined_deg, delta_nu, epsilon, config.numax)

    select_imgs = torch.Tensor(imgs).float().to(device).unsqueeze(0)
    select_numax = torch.Tensor([config.numax/1000.]).float().to(device)
    select_numax_sigma = torch.Tensor([config.numax_sig*100./config.numax]).float().to(device)
    select_teff = torch.Tensor([config.teff/5772.]).float().to(device)
    select_teff_sigma = torch.Tensor([config.teff_sig*100./config.teff]).float().to(device)
    select_fe_h = torch.Tensor([config.fe_h]).float().to(device)
    select_fe_h_sigma = torch.Tensor([config.fe_h_sig]).float().to(device)
    select_epsilon = torch.Tensor([epsilon]).float().to(device)
    select_delta_nu = torch.Tensor([WLS.coef_.squeeze()/100.]).float().to(device)

    select_pi,select_sigma,select_mu = trained_model(input_img=select_imgs,input_numax=select_numax, input_teff=select_teff, input_fe_h= select_fe_h,
                                                     input_numax_sigma= select_numax_sigma,
                input_teff_sigma= select_teff_sigma, input_fe_h_sigma=select_fe_h_sigma,
                                                     input_delta_nu=select_delta_nu,
                                                     input_epsilon=select_epsilon)

    output_params = ['age', 'mass', 'helium', 'metallicity', 'alpha', 'overshoot', 'diffusion',
'undershoot', 'radius', 'luminosity']
    output_pi, output_sigma, output_mu, output_grid, output_median, output_conf_interval, output_pdf = [], [], [], [], [], [], []


    print('--------- RESULTS ---------')
    for i, params in enumerate(output_params):
        param_pi = select_pi.data.cpu().numpy().squeeze()
        param_sigma = select_sigma[:,:,i].data.cpu().numpy().squeeze()
        param_mu = select_mu[:,:,i].data.cpu().numpy().squeeze()

        if params in ['metallicity', 'overshoot', 'undershoot', 'diffusion']:
            param_mu = np.exp(param_mu)
            param_sigma = np.exp(param_sigma)

        param_grid = np.arange(np.min(param_mu.squeeze()) - 10*param_sigma.squeeze()[np.argmin(param_mu.squeeze())], 
                         np.max(param_mu.squeeze()) + 10*param_sigma.squeeze()[np.argmax(param_mu.squeeze())], 0.0001)
        param_pdf = mix_pdf(param_grid, param_mu.squeeze(), param_sigma.squeeze(), param_pi.squeeze())
        param_cumsum_grid = np.cumsum(param_pdf)/(np.sum(param_pdf))
        param_quartile_vec = param_grid[np.argmin(np.abs(param_cumsum_grid-0.16))]
        param_median_vec = param_grid[np.argmin(np.abs(param_cumsum_grid-0.5))]
        param_third_quartile_vec = param_grid[np.argmin(np.abs(param_cumsum_grid-0.83))]
        param_mean= dist_mu_npy(param_pi.reshape(1,-1), param_mu.reshape(1,-1))
     

        print(params + ': ' + '%.3f +%.3f -%.3f' %(param_median_vec, param_third_quartile_vec - param_median_vec, param_median_vec - param_quartile_vec))

        output_pi.append(param_pi)
        output_sigma.append(param_sigma)
        output_mu.append(param_mu)
        output_grid.append(param_grid)
        output_median.append(param_median_vec)
        output_conf_interval.append([param_quartile_vec, param_third_quartile_vec])
        output_pdf.append(param_pdf)

    if config.output_10d:
        plot_10d(output_pi, output_mu, output_sigma, output_grid, output_median, output_conf_interval, output_pdf)

if __name__ == '__main__':
    infer(config)





