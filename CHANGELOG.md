## 0.1

API Changes

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
