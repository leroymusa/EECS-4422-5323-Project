UV-C Fluorescence Imaging Dataset — Saliva Residues on Surfaces
================================================================

Overview
--------
This dataset contains 10 RAW fluorescence images captured using a Sony NEX-6 camera that was modified for UV-C fluorescence imaging.
The purpose of this dataset is to analyze and model the fluorescence emission of saliva residues under UV-C excitation across different surface materials in both fresh (wet) and dried conditions.

Each RAW image corresponds to a unique experimental condition as described below.
Binary ground-truth masks and corresponding RGB reference images are also provided.

----------------------------------------------------------------
Imaging Setup
-------------
- Camera: Sony NEX-6 (full-spectrum modified)
- Lens: UV-transmissive lens with Baader U filter (320–380 nm bandpass)
- Bayer Pattern: RGGB (4 channels — 2×Green, 1×Red, 1×Blue)
- Illumination: UVC LED array, peak wavelength 265 nm, positioned 25 cm above the surface
- Exposure Settings:
  - Aperture: f/3.5
  - ISO: 100
  - Shutter Speed: 3.2 seconds
- Lighting Condition: Illumination solely from the UVC source (no ambient light)

You can visualize or extract the signal strength of individual channels using the RawPy Python library.

----------------------------------------------------------------
Dataset Composition
-------------------
There are 10 image pairs, each corresponding to a unique material and condition (fresh or dried saliva).
Images are numbered from 1 to 10 and follow this order:

1. Aluminum — Fresh
2. Aluminum — Dried
3. Steel — Fresh
4. Steel — Dried
5. Whitewood — Fresh
6. Whitewood — Dried
7. PVC — Fresh
8. PVC — Dried
9. Wood — Fresh
10. Wood — Dried

----------------------------------------------------------------
File Structure
--------------
dataset/
│
├── Raw_images/
│   ├── 1.ARW
│   ├── 2.ARW
│   └── ...
│
├── RGB_images/
│   ├── aluminum_rgb.jpg
│   ├── pvc_rgb.jpg
│   └── ...
│
├── Binary_masks/
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
└── README.txt

----------------------------------------------------------------
RGB Images
-----------
In addition to the UV-C fluorescence RAW images, each condition includes a corresponding RGB image captured under ambient light illumination using the same camera sensor.
These images serve as visual references to show the surface appearance and contamination visibility under normal lighting conditions.

----------------------------------------------------------------
Ground Truth Masks
------------------
Each fluorescence condition has a corresponding binary mask stored in the binary_masks/ directory.
- File naming follows the same numbering as the RAW images (1.png, 2.png, …).
- Mask pixel values:
  - 1 → foreground (saliva / fluorescent residue)
  - 0 → background

----------------------------------------------------------------
Citation
--------
If you use this dataset, please cite:
Akhavan, M. “Detection of Surface Contamination using UV-C Fluorescence Imaging,” M.A.Sc. Thesis, Prof. James Elder Lab, York University, 2025.
