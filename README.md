# PointNet++ MATLAB Toolbox

This repository contains a MATLAB toolbox for performing classification of 3D chemical field data using the PointNet++ architecture.

## Features
- Transform volumetric ECD data into point cloud (.ply)
- Perform classification using hierarchical PointNet++ model
- Includes demonstration on real molecular and object datasets

## Installation
Compatible with MATLAB R2024a.

## Instructions for Testing
To facilitate testing and ensure reproducibility, please follow the steps below:

1. **Download and organize the project folders**  
   Download the four main folders from GitHub: `Dataset-1`, `Dataset-2`, `Files01`, and `Files02`.  
   Place them in a **single parent directory** to ensure that all MATLAB scripts can access the correct paths.

2. **Run MATLAB and choose function modules**  
   Open **MATLAB R2024a** (or later).
   - The folder `Files01` contains scripts for converting 3D chemical field data into point cloud files (`.ply` format).
   - The folder `Files02` contains the classification scripts based on PointNet++.
   When testing different datasets (`Dataset-1` and `Dataset-2`), make sure to copy the internal folders named `train_big`, `val_big` and `test_external` into `Files02`, so that the classifier can access them using the correct directory structure.

3. **Ensure consistent global parameters**  
   Before running `Runme1_train.m` and `Runme2_postClassEval.m` in sequence, verify that the following global parameters are set to the same values in both scripts:  
   `Dim`, `kkk`, `nSAMP`, and `numPoints`.  
   Maintaining consistency ensures a fair evaluation of the model performance on the corresponding dataset.

For detailed instructions, refer to the script headers and in-line comments.

## Citation
If you use this toolbox, please cite:
> [Title], [Journal], [Year].

## MIT License
Copyright (c) 2025 Medical Intelligence Research Laboratory, Tongren University, China

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:                       

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.                                

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
