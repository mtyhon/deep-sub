# Mixture Density Network Parameters
The files in this folder contains the MDN parameters for the four main Kepler subgiants in the paper.


File format
---
Each file contains 16 rows corresponding to the 16 Gaussians used to estimate the output distribution. The first column (pi) is the mixing coefficient i.e. the weight of each Gaussian. The other columns correspond to each output parameter's mean (mu) and standard deviation (sigma) for a particular Gaussian. 


Creating a probability density for a parameter
---
Using the MDN parameters, a probability density is created by a linear combination of each Gaussian, weighted by their corresponding mixing coefficient. The following function does the trick:

```
    def mix_pdf(x, loc, scale, weights):
        d = np.zeros_like(x)
        count = 0
        for mu, sigma, pi in zip(loc, scale, weights):
            d += pi * norm.pdf(x, loc=mu, scale=sigma)
            count += 1
    return d 
```

Then, run the function:

```
prob_density = mix_pdf(parameter_grid, mu, sigma, pi)
```

and the variable `prob_density` will contain the probability density values over the range of parameter values defined by `parameter_grid`.



