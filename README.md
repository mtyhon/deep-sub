# deep-sub
Asteroseismic inference of subgiant star fundamental parameters using deep learning. This algorithm, as described by Hon et al. (submitted), is a convolutional mixture density network that uses oscillation mode frequencies, spectroscopic measurements, and global asteroseismic parameters to estimate a 10D distribution comprising the age, mass, radius, luminosity, initial helium abundance, initial metal abundance, diffusion multiplier, overshooting coefficient, and undershooting coefficient of a subgiant star. 


![alt text](https://github.com/mtyhon/deep-sub/raw/master/sample/contour_gemma.png "Gemma Age and Mass Estimated Distribution")



The default trained network provided in this repo was trained using MESA r12778 stellar models, and can be readily used to subgiant star observations of near 1-year length.

Future work will see this network extended to shorter time series, where detected oscillation modes are more sparse.

Required libraries:
---

* numpy
* scipy
* scikit-learn
* torch (>= 0.4.0)


Running the script
===


For parameter estimation, download the folder and run inference.py. The script accepts the following arguments:


* '--star_id': Identifier of the star of interest
* '--mode_filename': Path to ASCII file containing mode frequencies of star
* '--teff': Input effective temperature in K
* '--teff_sig': Effective temperature uncertainty in K
* '--numax': Frequency at maximum oscillation power in uHz
* '--numax_sig': Uncertainty for numax in uHz
* '--fe_h': Metallicity of star in decimal exponents (dex)
* '--fe_h_sig': Metallicity uncertainty in decimal exponents (dex)
* '--output_10d': Boolean flag telling the script to output a plot named "results.png" showing the distribution of each estimated parameter. Default is set to False.



The script outputs the median and credible intervals for each estimated parameter.
Optionally, set '--output_10d' to True to visualize the distribution of each parameter.

Running inference, an example command:
---

python inference.py --star_id 11026764 --mode_filename sample_gemma.dat --teff 5682 --teff_sig 84 --numax 890 --numax_sig 12 --fe_h 0.05 --fe_h_sig 0.09 --output_10d True


Input mode frequency file format
===

The example provided uses the provided mode frequencies sample_gemma.dat from the subgiant star Gemma, which was analyzed by Metcalfe et al. (2014) based on extracted frequencies by Appourchaux et al. (2012). 

The file containing mode frequencies has to be formatted with whitespace separation in the following manner:


| id       	| l   	| freq   	| freq_err 	|
|----------	|-----	|--------	|----------	|
| 11026764 	| 0   	| 674.57 	| 0.32     	|
| 11026764 	| 0   	| 723.52 	| 0.32     	|
| 11026764 	| 1   	| 799.54 	| 0.17     	|
| 11026764 	| 2   	| 818.63 	| 0.15     	|
| 11026764 	| ... 	| ...    	| ...      	|


See sample_gemma.dat for formatting reference.
