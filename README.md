# OpenEQA utilities
This repository contains utilities for the OpenEQA project. This is an independent project intended to pre-process the OpenEAQ data, explore more options, project easily in 3D using voxels maps, among other.

## News and Updates:
- **2025-04-19**: Version `v1.0.1`. Added pre-computation of scene coordinates. see [Scene Coordinates](https://github.com/EnriqueSolarte/open_eqa_utils/blob/e4c5202bf4edff2fb206894af81286927143fadc/examples/scene_coordinates/pre_computed_sc.py#L85).

<p align="center">
  <img src="./assets/scene_test.png"/>
</p>

## Installation
To install the package, you can use the following command:

```bash
conda create -n openeqa python=3.9
conda activate openeqa
git clone git@github.com:EnriqueSolarte/open_eqa_utils.git
cd open_eqa_utils
pip install -r requirements.txt
pip install -e .
```
