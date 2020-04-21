import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm
from torch.autograd import Variable
from torch.utils import data
from torch.distributions import Categorical

ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a Gaussian mixture probability distribution, where
    each Gaussian has num_gaussian dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxK, BxKxG, BxKxG): B is the batch size, K is the
            number of Gaussians, and G is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians, temperature=1):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.temperature = temperature
        self.pi = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,num_gaussians),
        )
        self.sigma = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,out_features*num_gaussians),
        )
        
        self.mu = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256,out_features*num_gaussians),
        )
        self.elu = nn.ELU()

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        pi = F.softmax(pi/self.temperature, dim=1)
        sigma = self.sigma(minibatch)
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)

        sigma[:, :, 0] = self.elu(sigma[:,:,0]) + 1 + 1e-6 # for Age
        sigma[:, :, 1] = self.elu(sigma[:,:,1]) + 1 + 1e-6# for Mass
        sigma[:, :, 2] = self.elu(sigma[:,:,2]) + 1 + 1e-6 # for Init Helium
        sigma[:, :, 4] = self.elu(sigma[:,:,4]) + 1 + 1e-6# for Alpha
        sigma[:, :, 8] = self.elu(sigma[:,:,8]) + 1 + 1e-6# for Radii
        sigma[:, :, 9] = self.elu(sigma[:,:,9]) + 1 + 1e-6# for Luminosity

        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


class Echelle_Conv2D_Array(nn.Module): # Incoming arrays SHOULD BE permuted to torch.Size([32, 1, 64, 128])
    def __init__(self, hidden_size, kernel_size, num_gaussians):
        super(Echelle_Conv2D_Array, self).__init__()

        self.kernel_size = kernel_size # in format of (height, width)
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,7), padding=(2, 3))  
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,5), padding=(1, 2)) 
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1,3), padding=(0,1))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(4096*9, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)

        
        self.mdn = MDN(in_features=self.hidden_size, out_features=10, num_gaussians=num_gaussians)

    def print_instance_name(self):
        print (self.__class__.__name__)

    def forward(self, input_img,input_numax, input_teff, input_fe_h,input_numax_sigma,
                input_teff_sigma, input_fe_h_sigma,input_delta_nu, input_epsilon):

        input_img = input_img.permute(0, 3, 2, 1)  # (N, C, H, W) so permute using (0,3,2,1)

        conv1 = F.leaky_relu(self.conv1(input_img), negative_slope=0.1)
        conv1 = self.pool1(conv1)

        conv2 = F.leaky_relu(self.conv2(conv1), negative_slope=0.1)
        conv2 = self.pool2(conv2)

        conv3 = F.leaky_relu(self.conv3(conv2), negative_slope=0.1)
        conv3 = self.pool3(conv3)

        conv3 = conv3.contiguous().view(conv3.size()[0], -1)
        input_features = torch.cat((torch.mul(conv3, input_numax.unsqueeze(-1).float()),torch.mul(conv3, input_numax_sigma.unsqueeze(-1).float()), torch.mul(conv3, input_teff.unsqueeze(-1).float()), torch.mul(conv3, input_teff_sigma.unsqueeze(-1).float()),torch.mul(conv3, input_fe_h.unsqueeze(-1).float()),torch.mul(conv3, input_fe_h_sigma.unsqueeze(-1).float()),torch.mul(conv3, input_delta_nu.unsqueeze(-1).float()),torch.mul(conv3, input_epsilon.float()), conv3),1)
        linear1 = F.relu(self.linear1(input_features))
        linear2 = F.relu(self.linear2(linear1))

        pi, sigma, mu = self.mdn(linear2)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given Gaussian mixture parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxKxG): The standard deviation of the Gaussians. B is the batch
            size, K is the number of Gaussians, and G is the number of
            dimensions per Gaussian.
        mu (BxKxG): The means of the Gaussians. B is the batch size, K is the
            number of Gaussians, and G is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxK): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    if len(sigma.size()) == 2:
        sigma = sigma.unsqueeze(-1) # need tensors to be 3d
        mu = mu.unsqueeze(-1)
    if len(target.size()) == 1:
        target = target.unsqueeze(-1)
    data = target.unsqueeze(1).expand_as(sigma)

    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((data - mu) / (sigma+1e-6))**2) / (sigma+1e-6)
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the Gaussian mixture parameters and the target
    The loss is the negative log likelihood of the data given the Gaussian mixture
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1)+1e-6)
    return torch.mean(nll)


def sample(pi, sigma, mu):
    """Draw samples from a Gaussian mixture.
    """
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i,idx]).add(mu[i,idx])
    return sample

def dist_mu(pi, mu):
    """Calculate the mean of a mixture.
    """
    if pi.size() != mu.size():
        pi = pi.unsqueeze(2)
    return torch.sum(pi*mu, dim=1)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred))


def initialization(model):
    for name, param in model.named_parameters():  # initializing model weights
        if 'bias' in name:
            nn.init.constant_(param, 0.00)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
        

def mix_pdf(x, loc, scale, weights):
    d = np.zeros_like(x)
    count = 0
    for mu, sigma, pi in zip(loc, scale, weights):
        d += pi * norm.pdf(x, loc=mu, scale=sigma)
        count += 1
    return d

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def dist_var_npy(pi, mu, mixture_mu, sigma):
    """Calculate the second moment (variance) of a bimodal distribution
    mu is the tensor while mixture_mu is the mean of the entire mixture as calculated by dist_mu
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    if mixture_mu.shape != mu.shape:
        mixture_mu = np.expand_dims(mixture_mu, -1)
    delta_square =(mu-mixture_mu)* (mu-mixture_mu)
    summation = sigma*sigma + delta_square
    return np.sum(pi*summation, 1)

def dist_var_npy_ltv(pi, mu, mixture_mu, sigma):
    """Calculate the second moment (variance) of a multimodal distribution using law of total variance
    mu is the tensor while mixture_mu is the mean of the entire mixture as calculated by dist_mu
    NOTE: This is identical to dist_var_npy, LTV is sum of the expectation of the variances with the variance of expectations
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    if mixture_mu.shape != mu.shape:
        mixture_mu = np.expand_dims(mixture_mu, -1)

    var = np.sum(pi*sigma*sigma, 1) + np.sum(pi*mu*mu, 1) - (mixture_mu**2).squeeze(1)

    return var


def dist_mu_npy(pi, mu):
    """Calculate the mean of a mixture.
    """
    if pi.shape != mu.shape:
        pi = np.expand_dims(pi, 2)
    return np.sum(pi*mu, 1)



