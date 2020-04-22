import os, torch, argparse, sys
sys.path.append("..") 
import torch.optim as optim
import torch.utils.data as utils
from model import *
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from utils.dataloader import d01_Dataset


parser = argparse.ArgumentParser()

parser.add_argument("--input_filename", type=str, required=True)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--kernel_size", type=int, default=5)
parser.add_argument("--num_gaussians", type=int,default=16)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--init_lr", type=float, default=0.001)
parser.add_argument("--num_epochs", type=int, default=500)
parser.add_argument("--save_best", type=bool, default=True)

config = parser.parse_args()

param_name = ['age','mass', 'helium', 'metallicity', 'alpha', 'overshoot', 'diffusion', 'undershoot', 'radii', 'luminosity']
input_tensor_indices = [0, 1, 2, 3, 6, 7, 8, 17, 18] 
param_tensor_indices = [4, 5, 9, 10, 11, 12, 13,14,15,16]

def validate(model, test_dataloader, device):

    model.eval() # set to evaluate mode

    for i, param in enumerate(param_name):
        exec("val_%s_cum_loss = 0" %param)
        exec("val_%s_cum_mape = 0" %param)
        exec("val_%s_cum_mae = 0" %param)
    
    total_val_entries = 0   
    val_batches = 0
    return_loss, return_mape, return_mae = [], [], []


    for j, val_data in tqdm(enumerate(test_dataloader, 0), total=len(test_dataloader), unit='batches'): 
        val_flag = val_data[-1].cuda()

        for k in range(len(val_data)):
            val_data[k] = val_data[k][val_flag == 0]

        input_tensor = [val_data[index].to(device).float() for index in input_tensor_indices]
        truth_tensor = [val_data[index].to(device).float() for index in param_tensor_indices]
        
        if (val_data[0].size()[0]) <= 1:
            print('Insufficient Batch!')
            continue
        if j == 3:
            break

        pi, sigma, mu = model(*input_tensor) 

        for exp_index in [3,5,6,7]:
            sigma[:, :, exp_index] = torch.exp(sigma[:, :, exp_index]) # exp log(mu) and log(sigma) for metallicity, overshoot, undershoot, diffusion
            mu[:, :, exp_index] = torch.exp(mu[:,:,exp_index])

        for i, param in enumerate(param_name):
            exec("val_%s_loss = mdn_loss(pi, sigma[:,:,i], mu[:,:,i], target=val_data[param_tensor_indices[i]].to(device).float())" %param)
            exec("val_%s_pred_npy = dist_mu(pi, mu[:,:,i]).data.cpu().numpy().reshape(-1,1)" %param)
            exec("val_%s_truth_npy = truth_tensor[i].float().data.cpu().numpy()" %(param))
            exec("val_%s_mape = mean_absolute_percentage_error(y_true=val_%s_truth_npy.squeeze(),y_pred=val_%s_pred_npy.squeeze())" %(param, param, param))  
            exec("val_%s_mae = mean_absolute_error(y_true=val_%s_truth_npy.squeeze(),y_pred=val_%s_pred_npy.squeeze())" %(param, param, param))                      
        
            exec("val_%s_cum_loss += val_%s_loss.item()" %(param,param))
            exec("val_%s_cum_mape += val_%s_mape.item()" %(param,param))
            exec("val_%s_cum_mae += val_%s_mae.item()" %(param,param))
          

        total_val_entries += val_data[0].size()[0]
        val_batches += 1


    for i, param in enumerate(param_name):
        exec("return_loss.append(val_%s_cum_loss/val_batches)"%param)
        exec("return_mape.append(val_%s_cum_mape/val_batches)"%param)  
        exec("return_mae.append(val_%s_cum_mae/val_batches)"%param) 

    return return_loss, return_mape, return_mae 


def train(model, model_optimizer, input_tensor, truth_tensor, device):

    return_mape, return_mae = [], [],
    model_optimizer.zero_grad()

    pi, sigma, mu = model(*input_tensor)


    for exp_index in [3,5,6,7]:
        sigma[:, :, exp_index] = torch.exp(sigma[:, :, exp_index]) # exp log(mu) and log(sigma) for metallicity, overshoot, undershoot, diffusion
        mu[:, :, exp_index] = torch.exp(mu[:,:,exp_index])

    # Calculate loss and backpropagate
    
    age_loss = mdn_loss(pi, sigma[:,:,0], mu[:,:,0], target=truth_tensor[0].to(device).float()) # log likelihood optim
    mass_loss = mdn_loss(pi, sigma[:,:,1], mu[:,:,1], target=truth_tensor[1].to(device).float())
    helium_loss = mdn_loss(pi, sigma[:,:,2], mu[:,:,2], target=truth_tensor[2].to(device).float())
    metallicity_loss = mdn_loss(pi, sigma[:,:,3], mu[:,:,3], target=truth_tensor[3].to(device).float())
    alpha_loss = mdn_loss(pi, sigma[:,:,4], mu[:,:,4], target=truth_tensor[4].to(device).float())
    overshoot_loss = mdn_loss(pi, sigma[:,:,5], mu[:,:,5], target=truth_tensor[5].to(device).float())
    diffusion_loss = mdn_loss(pi, sigma[:,:,6], mu[:,:,6], target=truth_tensor[6].to(device).float())
    undershoot_loss = mdn_loss(pi, sigma[:,:,7], mu[:,:,7], target=truth_tensor[7].to(device).float())
    radii_loss = mdn_loss(pi, sigma[:,:,8], mu[:,:,8], target=truth_tensor[8].to(device).float())
    luminosity_loss = mdn_loss(pi, sigma[:,:,9], mu[:,:,9], target=truth_tensor[9].to(device).float())

    #loss = 0
    #for i, param in enumerate(param_name):
    #    exec("%s_loss = mdn_loss(pi, sigma[:,:,i], mu[:,:,i], target=truth_tensor[i].to(device).float())" %param)
       
    loss = age_loss + mass_loss + helium_loss + 0.1*metallicity_loss + alpha_loss + 0.1*overshoot_loss + 0.1*diffusion_loss + 0.1*undershoot_loss + radii_loss + luminosity_loss
    loss.backward()
    #Clipnorm?
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update parameters
    model_optimizer.step()

    for i, param in enumerate(param_name):
        exec("%s_pred_mean = dist_mu(pi, mu[:,:,i]).data.cpu().numpy().reshape(-1,1)" %param)
        exec("%s_truth_npy = truth_tensor[i].float().data.cpu().numpy()" %param)
        exec("%s_mape = mean_absolute_percentage_error(y_true=%s_truth_npy.squeeze(), y_pred=%s_pred_mean.squeeze())" %(param, param, param))
        exec("%s_mae = mean_absolute_error(y_true=%s_truth_npy.squeeze(), y_pred=%s_pred_mean.squeeze())" %(param, param, param))
        exec("return_mape.append(%s_mape)" %param)
        exec("return_mae.append(%s_mae)" %param)

    return loss.item(), return_mape, return_mae


def load_grid(filename):

    data = np.load(filename, allow_pickle=True)['data']

    var_names = ['age', 'radii', 'mass', 'period_spacing', 'empirical_acoustic_cutoff', 'numax', 'teff', 'fe_h', 'luminosity', 'init_helium', 'init_metallicity', 'alpha', 'diffusion', 'overshoot', 'undershoot', 'freqs', 'deg', 'order', 'gravity_order', 'inertia']

    age = np.array([data[i][0] for i in range(len(data))])
    radii = np.array([data[i][1] for i in range(len(data))])
    mass = np.array([data[i][2] for i in range(len(data))])
    period_spacing = np.array([data[i][3] for i in range(len(data))])
    empirical_acoustic_cutoff = np.array([data[i][4] for i in range(len(data))])
    numax = np.array([data[i][5] for i in range(len(data))])
    teff = np.array([data[i][6] for i in range(len(data))])
    fe_h = np.array([data[i][7] for i in range(len(data))])
    luminosity = np.array([data[i][8] for i in range(len(data))])
    init_helium = np.array([data[i][9] for i in range(len(data))])
    init_metallicity = np.array([data[i][10] for i in range(len(data))])
    alpha = np.array([data[i][11] for i in range(len(data))])
    diffusion = np.array([data[i][12] for i in range(len(data))])
    overshoot = np.array([data[i][13] for i in range(len(data))])
    undershoot = np.array([data[i][14] for i in range(len(data))])
    freqs = np.array([data[i][15] for i in range(len(data))])
    deg = np.array([data[i][16] for i in range(len(data))])
    order = np.array([data[i][17] for i in range(len(data))])
    gravity_order = np.array([data[i][18] for i in range(len(data))])
    inertia = np.array([data[i][19] for i in range(len(data))])

    #for v, vname in enumerate(var_names):
    #    exec("%s = np.array([%s[i][v] for i in range(len(%s))])" %(vname, datax, datax))
 
    inertia = inertia/(4*np.pi) # proper normalization of inertia
    numax = numax/1000.
    teff = teff/5772.
    overshoot[overshoot == 0] = np.random.uniform(low=0.0, high = np.min(overshoot[overshoot != 0]), size = len(overshoot[overshoot == 0]))
    diffusion[diffusion == 0] = np.random.uniform(low=0.0, high = np.min(diffusion[diffusion != 0]), size = len(diffusion[diffusion == 0]))
    undershoot[undershoot == 0] = np.random.uniform(low=0.0, high = np.min(undershoot[undershoot != 0]), size = len(undershoot[undershoot == 0]))

    unique_masses, mass_indices = np.unique(mass, return_index=True)
    mass_train_indices, mass_test_indices = train_test_split(mass_indices, test_size=0.15, random_state=137)

    for v, vname in enumerate(var_names):
        exec("%s_train = %s[np.in1d(mass, mass[mass_train_indices])]" %(vname,vname))
        exec("%s_test = %s[np.in1d(mass, mass[mass_test_indices])]" %(vname,vname))

    dataloader_input_indices = [15, 16, 17, 5, 6, 7, 2, 0, 4, 19, 9, 10, 11, 13, 12, 14, 1, 8]
    dataloader_train_params, dataloader_test_params = [], []

    for ix in dataloader_input_indices:
        if var_names[ix] == 'numax':
            exec("dataloader_train_params.append(%s_train*1000.)" %var_names[ix])
            exec("dataloader_test_params.append(%s_test*1000.)" %var_names[ix])
        else:
            exec("dataloader_train_params.append(%s_train)" %var_names[ix])
            exec("dataloader_test_params.append(%s_test)" %var_names[ix])

    #freqs,deg,order,numax,teff,fe_h,mass,age,acoustic_cutoff,inertia,init_helium, init_metallicity,alpha, overshoot, diffusion, undershoot,radii, luminosity

    train_dataset = d01_Dataset(*dataloader_train_params)
    train_dataloader = utils.DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size, num_workers=15)

    test_dataset = d01_Dataset(*dataloader_test_params)
    test_dataloader = utils.DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size, num_workers=15)

    return train_dataloader, test_dataloader


def full_training(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = Echelle_Conv2D_Array(hidden_size=config.hidden_size, kernel_size=config.kernel_size, num_gaussians=config.num_gaussians)
    model.to(device)

    print(str(model))

    learning_rate = 0.001
    model_optimizer = optim.Adam(model.parameters(), lr = config.init_lr)
    scheduler = ReduceLROnPlateau(model_optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr = 1E-6)
    best_loss = 10000
    best_mape = 10000

    initialization(model)

    n_epochs = config.num_epochs
    model_checkpoint = config.save_best
    train_dataloader, test_dataloader = load_grid(config.input_filename)
 
    for epoch in range(1, n_epochs + 1):
        print('---------------------')
        print('Epoch: ', epoch)

        train_loss = 0
        train_batches = 0

        for i, param in enumerate(param_name):
            exec("%s_cum_mape = 0" %param)
            exec("%s_cum_mae = 0" %param)            

        model.train()  # set to training mode


        for i, data in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), unit='batches'): 
            train_batches += 1
            train_flag = data[-1].cuda()

            for m in range(len(data)):
                data[m] = data[m][train_flag == 0]

            if (data[0].size()[0]) <= 1:
                print('Insufficient Batch!')
                continue

            input_tensor = [data[index].to(device).float() for index in input_tensor_indices]
            truth_tensor = [data[index].to(device).float() for index in param_tensor_indices]
            
            loss, mape, mae = train(model, model_optimizer, input_tensor, truth_tensor, device)
            if i == 3:
                break
            train_loss += loss

            for i, param in enumerate(param_name):
                exec("%s_cum_mape += mape[i]" %param)
                exec("%s_cum_mae += mae[i]" %param)

        train_loss = train_loss/train_batches

        val_loss, val_mape, val_mae  = validate(model, test_dataloader, device)
        total_val_loss=np.sum(np.array([vloss for vloss in val_loss]))
        scheduler.step(val_loss[0]) # reduce LR on loss plateau

        print('---LOSSES---')
        print('Train Loss: ', train_loss)
        print('Val Loss: ', total_val_loss)

        for i, param in enumerate(param_name):
            exec("print('---%s Metrics---')" %param)
            exec("print('%s Train MAPE: ', %s_cum_mape/train_batches)" %(param, param))
            exec("print('%s Val MAPE: ', val_mape[i])" %(param))
            exec("print('%s Train MAE: ', %s_cum_mae/train_batches)" %(param, param))
            exec("print('%s Val MAE: ', val_mae[i])" %(param))


        for param_group in model_optimizer.param_groups:
            print('Current Learning Rate: ', param_group['lr'])

        is_best = val_mape[0] < best_mape
        best_mape = min(val_mape[0], best_mape)
        print('Current Best Metric: ', best_mape)

        if config.save_best:
            if is_best:
                best_loss = total_val_loss
                filename = '%d_Loss:%.3f.torchmodel' % (epoch, best_loss)
                package_dir = os.path.dirname(os.path.abspath(__file__))
                filepath = package_dir + '/checkpoint/'

                torch.save(model.state_dict(), os.path.join(filepath, filename))
                print('Model saved to %s' %os.path.join(filepath, filename))
                
            else:
                print('No improvement over the best of %.4f' %best_loss)

if __name__ == '__main__':
    full_training(config)


