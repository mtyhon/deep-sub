from torch.utils import data
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure
from scipy.ndimage import binary_dilation


class d01_Dataset(data.Dataset):
    # On the fly freqs generation for training. This version is repurposed for MDN
    # This method randomizes the sigma for auxiliary variables, perturbes the variables, and returns sigma as an input
    def __init__(self, freqs,deg,order,numax,teff,fe_h,mass,age,acoustic_cutoff,inertia,init_helium, init_metallicity,alpha, overshoot, diffusion, undershoot,radii, luminosity, 
                 perturb_freqs=False, perturb_teff=False, perturb_numax=False, 
                 perturb_fe_h=False, include_quad=False, stochastic_sampling=False):
        self.freqs = freqs
        self.order =order
        self.deg = deg
        self.numax = numax
        self.teff = teff
        self.fe_h = fe_h
        self.init_helium = init_helium
        self.init_metallicity = init_metallicity
        self.alpha = alpha
        self.overshoot = overshoot
        self.diffusion = diffusion
        self.undershoot = undershoot
        self.luminosity = luminosity
        self.radii = radii
        self.age = age
        self.mass = mass
        self.acoustic_cutoff = acoustic_cutoff
        self.inertia = inertia
        self.stochastic_sampling = stochastic_sampling

        self.perturb_freqs = perturb_freqs
        self.perturb_numax = perturb_numax
        self.perturb_teff = perturb_teff
        self.perturb_fe_h = perturb_fe_h
        self.include_quad = include_quad

    def __len__(self):
        'Total number of samples'
        return len(self.freqs)

    def __getitem__(self, index):
        'Generates ONE sample of data'
        batch_freqs = self.freqs[index]
        batch_deg = self.deg[index]
        batch_order = self.order[index]
        batch_numax = self.numax[index]
        batch_teff = self.teff[index]
        batch_fe_h = self.fe_h[index]
        batch_age = self.age[index]
        batch_mass = self.mass[index]
        batch_acoustic_cutoff = self.acoustic_cutoff[index]
        batch_inertia = self.inertia[index]
        batch_init_helium = self.init_helium[index]
        batch_init_metallicity = self.init_metallicity[index]
        batch_alpha = self.alpha[index]
        batch_overshoot = self.overshoot[index]
        batch_undershoot = self.undershoot[index]
        batch_diffusion = self.diffusion[index]
        batch_radii = self.radii[index]
        batch_luminosity = self.luminosity[index]
        
        if self.perturb_numax:
            batch_numax, batch_numax_sigma= self.__perturb_numax(numax=batch_numax)
        else:
            batch_numax_sigma = np.zeros(batch_numax.shape)
        if self.perturb_teff:
            batch_teff, batch_teff_sigma= self.__perturb_teff(teff=batch_teff)
        else:
            batch_teff_sigma = np.zeros(batch_teff.shape)
        if self.perturb_fe_h:
            batch_fe_h, batch_fe_h_sigma= self.__perturb_fe_h(fe_h=batch_fe_h)
        else:
            batch_fe_h_sigma = np.zeros(batch_fe_h.shape)
        
        batch_img, batch_delta_nu, batch_epsilon, batch_flag, nb_freqs_used, nb_radial_freqs_used, nb_dipole_freqs_used, nb_quad_freqs_used = self.__data_generation(batch_freqs, batch_deg, batch_order, batch_numax, batch_inertia, batch_acoustic_cutoff)
        return batch_img, batch_numax/1000.,batch_teff, batch_fe_h,batch_age, batch_mass, batch_numax_sigma,batch_teff_sigma, batch_fe_h_sigma, batch_init_helium, batch_init_metallicity, batch_alpha, batch_overshoot, batch_diffusion, batch_undershoot, batch_radii, batch_luminosity,batch_delta_nu/100., batch_epsilon, nb_freqs_used, nb_radial_freqs_used, nb_dipole_freqs_used, nb_quad_freqs_used, batch_flag


    def __data_generation(self, batch_freqs, batch_deg, batch_order, batch_numax, batch_inertia, batch_acoustic_cutoff):
        selection_freq = batch_freqs[batch_freqs != 0]
        selection_deg = batch_deg[batch_freqs != 0]
        selection_order = batch_order[batch_freqs != 0]
        selection_inertia = batch_inertia[batch_freqs != 0]
        
        selection_freq = selection_freq[~np.isnan(selection_inertia)]
        selection_deg = selection_deg[~np.isnan(selection_inertia)]
        selection_order = selection_order[~np.isnan(selection_inertia)]
        selection_inertia = selection_inertia[~np.isnan(selection_inertia)]


        radial_freqs = selection_freq[selection_deg == 0]
        radial_order = selection_order[selection_deg == 0]
        radial_deg = selection_deg[selection_deg == 0]
        radial_inertia = selection_inertia[selection_deg == 0]
        dipole_freqs = selection_freq[selection_deg == 1]
        dipole_order = selection_order[selection_deg == 1]
        dipole_deg = selection_deg[selection_deg == 1]
        dipole_inertia = selection_inertia[selection_deg == 1]
        quad_freqs = selection_freq[selection_deg == 2]
        quad_order = selection_order[selection_deg == 2]
        quad_deg = selection_deg[selection_deg == 2]
        quad_inertia = selection_inertia[selection_deg == 2]

        if self.perturb_freqs:
            radial_freqs, dipole_freqs, quad_freqs, radial_perturb_magnitude, dipole_perturb_magnitude = self.__perturb_freqs(radial_freqs, dipole_freqs, quad_freqs)

        try:
            radial_closest = np.sort(radial_freqs[np.argsort(np.abs(radial_freqs - batch_numax))[:6]])
            order_closest = np.sort(radial_order[np.argsort(np.abs(radial_freqs - batch_numax))[:6]])
        except:
            radial_closest = np.sort(radial_freqs[np.argsort(np.abs(radial_freqs - batch_numax))[:]])
            order_closest = np.sort(radial_order[np.argsort(np.abs(radial_freqs - batch_numax))[:]])    
     
        fwhm = 2*np.sqrt(2*np.log(2))
        weights = sp.stats.norm.pdf(radial_closest, batch_numax, 0.25*batch_numax/fwhm) # 0.25 is default
        WLS_init = LinearRegression()
        WLS_init.fit(order_closest.reshape(-1, 1), radial_closest.reshape(-1, 1), sample_weight=weights)
        init_delta_nu = WLS_init.coef_.squeeze()
        

        combined_freqs = np.concatenate([radial_freqs, dipole_freqs])
        combined_freqs = np.append(combined_freqs, quad_freqs) 
        combined_order = np.concatenate([radial_order, dipole_order]) 
        combined_order = np.append(combined_order, quad_order) 
        combined_deg = np.concatenate([radial_deg, dipole_deg]) 
        combined_deg = np.append(combined_deg, quad_deg)
        combined_inertia = np.concatenate([radial_inertia, dipole_inertia]) 
        combined_inertia = np.append(combined_inertia, quad_inertia)

        combined_inertia = combined_inertia[np.argsort(combined_freqs)]
        combined_deg = combined_deg[np.argsort(combined_freqs)]
        combined_order = combined_order[np.argsort(combined_freqs)]
        combined_freqs = combined_freqs[np.argsort(combined_freqs)]

        if self.stochastic_sampling:
            if self.include_quad:
                combined_freqs_dipole = combined_freqs[combined_deg == 1]
                combined_deg_dipole = combined_deg[combined_deg == 1]
                combined_order_dipole = combined_order[combined_deg == 1]
                combined_inertia_dipole = combined_inertia[combined_deg == 1]
                combined_freqs_radial = combined_freqs[combined_deg == 0]
                combined_deg_radial = combined_deg[combined_deg == 0]
                combined_order_radial = combined_order[combined_deg == 0]
                combined_inertia_radial = combined_inertia[combined_deg == 0]
                combined_freqs_quad = combined_freqs[combined_deg == 2]
                combined_deg_quad = combined_deg[combined_deg == 2]               
                combined_order_quad = combined_order[combined_deg == 2]   
                combined_inertia_quad = combined_inertia[combined_deg == 2]
                                                         
                delta_nu_range_dipole = np.random.randint(low=4, high=8, size=1)[0] # low=4; sparse :- low=2
                dipole_limit_filter = (combined_freqs_dipole >= (batch_numax - delta_nu_range_dipole*init_delta_nu)) & (combined_freqs_dipole <= (batch_numax + delta_nu_range_dipole*init_delta_nu))

                delta_nu_range_radial = np.random.randint(low=4, high=8, size=1)[0] # low=4; sparse :- low=1
                radial_limit_filter = (combined_freqs_radial >= (batch_numax - delta_nu_range_radial*init_delta_nu)) & (combined_freqs_radial <= (batch_numax + delta_nu_range_radial*init_delta_nu))    
                if delta_nu_range_radial == 1:
                    delta_nu_range_quad = 1
                else:
                    delta_nu_range_quad = np.random.randint(low=3, high=delta_nu_range_radial, size=1)[0] # low=3; sparse :- low=1
                quad_limit_filter = (combined_freqs_quad >= (batch_numax - delta_nu_range_quad*init_delta_nu)) & (combined_freqs_quad <= (batch_numax + delta_nu_range_quad*init_delta_nu)) 

                combined_freqs_dipole = combined_freqs_dipole[dipole_limit_filter]
                combined_deg_dipole = combined_deg_dipole[dipole_limit_filter]
                combined_order_dipole = combined_order_dipole[dipole_limit_filter]
                combined_inertia_dipole = combined_inertia_dipole[dipole_limit_filter]
                combined_freqs_radial = combined_freqs_radial[radial_limit_filter]
                combined_deg_radial = combined_deg_radial[radial_limit_filter]
                combined_order_radial = combined_order_radial[radial_limit_filter]
                combined_inertia_radial = combined_inertia_radial[radial_limit_filter]
                combined_freqs_quad = combined_freqs_quad[quad_limit_filter]
                combined_deg_quad = combined_deg_quad[quad_limit_filter]               
                combined_order_quad = combined_order_quad[quad_limit_filter]   
                combined_inertia_quad = combined_inertia_quad[quad_limit_filter]
                
                candidate_freqs_quad, candidate_deg_quad, candidate_order_quad, candidate_inertia_quad = [], [], [], []
 
                for idx in np.unique(combined_order_quad):
                    order_select = combined_order_quad[combined_order_quad == idx]
                    freq_select = combined_freqs_quad[combined_order_quad == idx]
                    deg_select = combined_deg_quad[combined_order_quad == idx]
                    inertia_select = combined_inertia_quad[combined_order_quad == idx]

                    candidate_freqs_quad.append(freq_select[np.argmin(inertia_select)])
                    candidate_deg_quad.append(deg_select[np.argmin(inertia_select)])
                    candidate_order_quad.append(order_select[np.argmin(inertia_select)])
                    candidate_inertia_quad.append(inertia_select[np.argmin(inertia_select)])

                combined_freqs_quad, combined_deg_quad, combined_order_quad, combined_inertia_quad = np.array(candidate_freqs_quad), np.array(candidate_deg_quad), np.array(candidate_order_quad), np.array(candidate_inertia_quad)
                
                combined_inertia = np.concatenate((combined_inertia_radial, combined_inertia_dipole, combined_inertia_quad))
                combined_deg = np.concatenate((combined_deg_radial, combined_deg_dipole, combined_deg_quad))
                combined_order = np.concatenate((combined_order_radial, combined_order_dipole, combined_order_quad))
                combined_freqs = np.concatenate((combined_freqs_radial, combined_freqs_dipole, combined_freqs_quad))
                
                combined_inertia = combined_inertia[np.argsort(combined_freqs)]
                combined_deg = combined_deg[np.argsort(combined_freqs)]
                combined_order = combined_order[np.argsort(combined_freqs)]
                combined_freqs = combined_freqs[np.argsort(combined_freqs)]

            else:
                delta_nu_range = np.random.randint(low=4, high=8, size=1)[0]
                limit_filter = (combined_freqs >= (batch_numax - delta_nu_range*init_delta_nu)) & (combined_freqs <= (batch_numax + delta_nu_range*init_delta_nu))
                combined_inertia = combined_inertia[limit_filter]                
                combined_deg = combined_deg[limit_filter]                
                combined_order = combined_order[limit_filter]                
                combined_freqs = combined_freqs[limit_filter]
            if len(combined_freqs[combined_deg == 0]) < 2:
                if self.include_quad:
                    imgs = np.zeros((128,64,3))
                else:
                    imgs = np.zeros((128,64,1))
                flag = 1
                return imgs, 0, np.array([0.0]), flag, 0, 0, 0, 0

            assert (len(combined_freqs[combined_deg == 0]) >= 2), batch_freqs

            combined_freqs = self.__correct_surface_effect(freqs=combined_freqs, acoustic_cutoff=batch_acoustic_cutoff, inertia=combined_inertia, spherical_deg = combined_deg, numax=batch_numax)

            combined_radial = combined_freqs[combined_deg == 0]
            combined_radial_order = combined_order[combined_deg == 0]

            radial_closest_combine = np.sort(combined_radial[np.argsort(np.abs(combined_radial - batch_numax))[:6]])
            order_closest_combine = np.sort(combined_radial_order[np.argsort(np.abs(combined_radial - batch_numax))[:6]])
            weights_combine = sp.stats.norm.pdf(radial_closest_combine, batch_numax, 0.25*batch_numax/fwhm) 
            WLS = LinearRegression()
            WLS.fit(order_closest_combine.reshape(-1, 1), radial_closest_combine.reshape(-1, 1), sample_weight=weights_combine)
            delta_nu = WLS.coef_.squeeze()
            epsilon = WLS.intercept_/delta_nu
            if epsilon >= 1:
                epsilon = epsilon % 1

            drop_weights = 0.95 # 5% chance to randomly drop modes from echelle

            roll  = np.random.uniform(size=len(combined_freqs))
            combined_order = combined_order[roll <= drop_weights]
            combined_deg = combined_deg[roll <= drop_weights]
            combined_freqs = combined_freqs[roll <= drop_weights]
        else:
            raise ValueError
            pass

        dipole_freqs_used = combined_freqs[combined_deg ==1]
        nb_freqs_used = len(dipole_freqs_used[(dipole_freqs_used >= batch_numax-7*delta_nu) & ((dipole_freqs_used <= batch_numax+7*delta_nu))])
        nb_radial_freqs_used = len(combined_freqs[combined_deg ==0])
        nb_dipole_freqs_used = len(combined_freqs[combined_deg ==1])
        nb_quad_freqs_used = len(combined_freqs[combined_deg ==2])    
        

        assert len(combined_freqs) == len(combined_order) == len(combined_deg), 'Incorrect Array Length!'
        
        flag = 0

        dimx = 128
        dimy = 64
        imgs = []
        if self.include_quad:
            degs = [0,1,2]
        else:
            degs = [1]
        imgs_array = np.empty((dimx, dimy, len(degs)))
        for k, ell in enumerate(degs):
            nu = combined_freqs[combined_deg == ell]
           
            reduced_freqs = (nu % delta_nu) - epsilon*delta_nu
            binx = np.linspace(-delta_nu, delta_nu, dimx+1).squeeze()
            biny = np.linspace(batch_numax-7*delta_nu, batch_numax+7*delta_nu, dimy+1).squeeze()
            reduced_freqs_extended = np.concatenate([reduced_freqs-delta_nu, reduced_freqs, delta_nu+reduced_freqs])
            nu_extended = np.concatenate([nu,nu,nu])
            nu_extended = nu_extended[(reduced_freqs_extended <=delta_nu)&(reduced_freqs_extended >= -delta_nu)]
            reduced_freqs_extended = reduced_freqs_extended[(reduced_freqs_extended <= delta_nu)&(reduced_freqs_extended >= -delta_nu)]
            try:                
                img, xedge, yedge, binnumber = binned_statistic_2d(reduced_freqs_extended, nu_extended, None, 'count', bins=[binx,biny])
            except:
                flag = 1
                img = np.zeros((dimx, dimy))
            st = generate_binary_structure(2, 2) # square expansion
            dilated = grey_dilation(img, footprint = iterate_structure(st,1), mode='constant') # arg 2 = repeat twice, 1=3x3; 2=5x5
            imgs_array[:,:,k] = dilated

            imgs +=[imgs_array]


        if self.include_quad:
            try:
                imgs = np.array(imgs)
                imgs = np.maximum(np.maximum(imgs[0], imgs[1]), imgs[2])

            except:
                imgs = np.zeros((128,64,3))
                flag = 1
        else:
            try:
                imgs = np.array(imgs)
                imgs = np.squeeze(imgs, 0) # if dipole only
            except:
                imgs = np.zeros((128,64,1))
                flag = 1
                
        return imgs, delta_nu, epsilon, flag, nb_freqs_used, nb_radial_freqs_used, nb_dipole_freqs_used, nb_quad_freqs_used

    def __perturb_freqs(self, radial_freqs, dipole_freqs, quad_freqs):
        radial_perturb_magnitude = np.random.uniform(low=0.1, high=1.)
        dipole_perturb_magnitude = np.random.uniform(low=0.5*radial_perturb_magnitude, high=1.*radial_perturb_magnitude)
        quad_perturb_magnitude = np.random.uniform(low=1.*radial_perturb_magnitude, high=2.*radial_perturb_magnitude)

        radial_freq_perturbations = np.random.normal(loc=0, scale=radial_perturb_magnitude, size=radial_freqs.shape) # l = 1 uncertainty is about 0.75 of l = 0?
        dipole_freq_perturbations = np.random.normal(loc=0, scale=dipole_perturb_magnitude, size=dipole_freqs.shape)
        quad_freq_perturbations = np.random.normal(loc=0, scale=quad_perturb_magnitude, size=quad_freqs.shape)
        return radial_freqs+radial_freq_perturbations, dipole_freqs+dipole_freq_perturbations, quad_freqs + quad_freq_perturbations, radial_perturb_magnitude, dipole_perturb_magnitude


    def __perturb_teff(self, teff):
        teff_sigma = np.random.randint(low=50, high=150, size=teff.shape)/5772.
        teff_sigma_fraction = teff_sigma*100./teff
        teff_perturbations = np.random.normal(loc=0, scale=teff_sigma, size=teff.shape)
        teff_perturbations_2 = np.random.normal(loc=0, scale=teff_sigma, size=teff.shape)

        return teff+teff_perturbations, teff_sigma_fraction

    def __perturb_numax(self, numax):
        numax_sigma_fraction = np.random.uniform(low=0.25, high=2.5, size=numax.shape)# 0.25 to 2.5%
        numax_sigma = numax_sigma_fraction*numax/100.
        numax_perturbations = np.random.normal(loc=0, scale=numax_sigma, size=numax.shape)
        numax_perturbations_2 = np.random.normal(loc=0, scale=numax_sigma, size=numax.shape)

        return numax+numax_perturbations, numax_sigma_fraction

    def __perturb_fe_h(self, fe_h):
        fe_h_sigma = np.random.randint(low=5, high=15, size=fe_h.shape)/100.
        fe_h_perturbations = np.random.normal(loc=0, scale=fe_h_sigma, size=fe_h.shape)
        fe_h_perturbations_2 = np.random.normal(loc=0, scale=fe_h_sigma, size=fe_h.shape)

        return fe_h+fe_h_perturbations, fe_h_sigma

    def __correct_surface_effect(self, freqs, acoustic_cutoff, inertia, spherical_deg, numax): # new formulation
        surf_radial_freqs = freqs[spherical_deg == 0]
        surf_radial_inertia = inertia[spherical_deg == 0]
        radial_at_numax = surf_radial_freqs[np.argmin(np.abs(surf_radial_freqs - numax))]
        inertia_at_numax = surf_radial_inertia[np.argmin(np.abs(surf_radial_freqs - numax))]
        upper_bound_correction = (0.38/100)*numax # maximum perturb value at high end of frequency
        lower_bound_correction = (0.22/100)*numax
        # What are the range of coefficients required to bring sample realistic corrections to the frequency?
        upper_coeff = upper_bound_correction*inertia_at_numax*np.power((acoustic_cutoff/radial_at_numax),3)
        lower_coeff = lower_bound_correction*inertia_at_numax*np.power((acoustic_cutoff/radial_at_numax),3)
        cubic_coeff = np.random.uniform(low=lower_coeff, high = upper_coeff, size=1)
        if np.isnan(lower_coeff):
            print('Freqs: ', freqs)
            print('Inertia: ', inertia)
            print('Upper Bound Correction: ', upper_bound_correction)
            print('Min Inertia: ', np.min(inertia))
            print('Lower Coeff: ', lower_coeff)
            print('Upper Coeff: ', upper_coeff)

        cubic_correction = cubic_coeff*((freqs/acoustic_cutoff)**3)/inertia
        correction = cubic_correction
        corrected_freqs = freqs - correction # model frequencies OVERESTIMATE real frequencies, so to correct models we subtract a correction from them.
       
        return corrected_freqs
