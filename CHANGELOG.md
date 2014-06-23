## 0.1

API Changes

  - An `init_weights` keyword argument has been added to the `get_biro_pipeline`. Possible values are `None`, `'sigma'` and `'weight'`. Respectively, these options govern, not initialising the weight vector or, initialising it from either the 'SIGMA' or 'WEIGHT' MeasurementSet columns.
  - The `noise_vector` keyword argument changes to the `weight_vector` keyword argument on the `get_biro_pipeline` method.
  - `transfer_noise_vector` changes to `transfer_weight_vector` on the `GPUSharedData` object.
  - The `noise_vector` is renamed to `weight_vector`. Additionally, the dimension of this matrix changes from (nbl, nchan, ntime) to (4, nbl, nchan, ntime).
  - The dimension of the `brightness` matrix changes from (5,nsrc) to (5,ntime,nsrc)
  - On the `BaseSharedData` class
    - The `nsrc` parameter changes to `npsrc` (Number of *Point* Sources).
    - An `ngsrc` (Number of *Guassian* Sources has been added).
    - `nsrc` now refers to the number of both gaussian and point sources.
  - `transfer_gauss_shape` added to the `GPUSharedData` object.
  - `montblanc.get_biro_pipeline`
    - Added a *noise_vector* argument, indicating whether to use the noise_vector to calculate the chi squared value.
  - `transfer_noise_vector` added to the `GPUSharedData` object.
  - `transfer_bayes_model` changes to `transfer_bayes_data` on `GPUSharedData` object.
  - `set_ref_freq changes` to `set_ref_wave` on `BaseSharedData` class.
  - `ref_freq` member changes to `ref_wave` on BaseSharedData` class.
  - `set_cos3_constant` has changed to `set_beam_width` on `BaseSharedData` class.
  - `cos3_constant` member changes to `beam_width` on `BaseSharedData` class.
