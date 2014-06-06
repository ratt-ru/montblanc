## 0.1

API Changes

  - `montblanc.get_biro_pipeline`
    - Added a *noise_vector* argument, indicating whether to use the noise_vector to calculate the chi squared value.
  - `transfer_noise_vector` added to the `GPUSharedData` object.
  - `transfer_bayes_model` changes to `transfer_bayes_data` on `GPUSharedData` object.
  - `set_ref_freq changes` to `set_ref_wave` on `BaseSharedData` class.
  - `ref_freq` member changes to `ref_wave` on BaseSharedData` class.
  - `set_cos3_constant` has changed to `set_beam_width` on `BaseSharedData` class.
  - `cos3_constant` member changes to `beam_width` on `BaseSharedData` class.
