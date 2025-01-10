Cortical gradients of neural dynamics 
----------------------------------------------

![Alt text](figures/title_fig.png?raw=true "Title")

Welcome to the code repository for our paper "Human cortical dynamics reflect graded contributions of local geometry and network topography"! We provide scripts and processed data files to reproduce the main analyses presented in the paper.

You can find our preprint here: 
https://www.biorxiv.org/content/10.1101/2025.01.07.631564v1 

### Key scripts

**generate_ieeg_gradients.m**
- Compute power spectrum density from iEEG timeseries
- Map channels to surface and parcellation
- Generate gradients

**gradients_and_networks.m**
- Compute embedding distance matrix
- Run independent correlations for each MRI modality

**gradients_multilinear.m**
- Fit multilinear model across all edges
- Fit region-specific multilinear model 


### Dependencies

The code in this repository depends on the following software. Please note the code has been developed and tested using MATLAB2021b.
- BrainSpace (https://github.com/MICA-MNI/BrainSpace): Gradient analysis
- Gifti matlab (https://github.com/gllmflndn/gifti)
- plotSurfaceROIBoundary (https://github.com/StuartJO/plotSurfaceROIBoundary): For visualization

The code furthermore relies on several open datasets. Processed versions of necessary data are provided in the repository.
- MNI Open iEEG atlas (https://ieegatlas.loris.ca/)
- MICA-MICs (https://osf.io/j532r/)
- MICA-PNI (https://osf.io/mhq3f/): Used for validation


### Contact
For questions regarding the code or analysis pipeline, please get in touch: jessica[dot]royer[at]mail[dot]mcgill[dot]ca

