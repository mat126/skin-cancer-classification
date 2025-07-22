# 🧬 ISIC 2024 Dataset – Variable Description

This file describes the structure and meaning of each variable used in the ISIC 2024 skin cancer classification dataset.

## 🔑 Identifiers and Demographics

- **isic_id**: Unique identifier for each image.
- **patient_id**: Anonymous identifier for the patient.
- **age_approx**: Approximate age of the patient (in years).
- **sex**: Sex of the patient – values: `male`, `female`, or `unknown`.

## 🩺 Clinical and Diagnostic Information

- **target**: Binary variable indicating malignancy (`1`) or benign (`0`).
- **lesion_id**: Identifier for the lesion (can be linked across images).
- **anatom_site_general**: General anatomical location of the lesion.
- **clin_size_long_diam_mm**: Clinically estimated maximum diameter (mm).
- **iddx_full**, **iddx_1–iddx_5**: Coded diagnosis levels.
- **mel_mitotic_index**: Melanoma mitotic index (mitoses per mm²).
- **mel_thick_mm**: Melanoma thickness (in mm).

## 🖼️ Image-Derived TBP Features

Variables derived from automated image processing (e.g., Lesion Visualizer):

- **tbp_tile_type**: Type of tile in Total Body Photography context.
- **tbp_lv_L, A, B, C, H, Lext, Aext, ...**: Average and extended values in CIELAB and HSV color spaces.
- **tbp_lv_deltaL, deltaA, ...**: Color differences between lesion and surrounding skin.
- **tbp_lv_color_std_mean**: Standard deviation of colors within the lesion.
- **tbp_lv_areaMM2, perimeterMM, area_perim_ratio**: Geometric features.
- **tbp_lv_eccentricity**: Eccentricity of the lesion.
- **tbp_lv_minorAxisMM**: Minor axis of fitted ellipse.
- **tbp_lv_symm_2axis, symm_2axis_angle**: Symmetry metrics.
- **tbp_lv_norm_border, norm_color**: Normalized irregularity and color variation.
- **tbp_lv_radial_color_std_max**: Max radial color deviation.
- **tbp_lv_x, y, z**: Spatial coordinates of lesion.
- **tbp_lv_location, location_simple**: Detailed and simplified location.
- **tbp_lv_nevi_confidence**: Confidence in identifying nevi.
- **tbp_lv_dnn_lesion_confidence**: Confidence of neural network prediction.

## 📎 Attribution and Licensing

- **attribution**: Image source or contributor.
- **copyright_license**: Image license (e.g., CC BY-NC 4.0).
