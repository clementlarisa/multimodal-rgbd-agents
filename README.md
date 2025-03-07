# Multimodal RGBD agents (Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents)  

> **Note**: This repository contains selected code related to our submitted paper **"Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents"**, currently under review for **IROS 2025**.  

## Description  
This repository includes **code for model definition and training**, but does **not** include:  
- Datasets  
- Code for loading/preprocessing data  

If you are interested in obtaining access to the datasets or data-loading scripts, please **contact us directly**.

## Paper  
- **Title**: "Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents"  
- **Authors**: Mihaela-Larisa Clement*, Monika Farsang*, Felix Resch, Radu Grosu
- **Conference**: Under review at IROS, 2025

## Repository Contents  
- `model_builder.py` – Model architectures for RGB, early and late fusion
- `deformable_model_builder.py` - Model architectures for depth-adaptive fusion
- `train.py` – Training script  
- `evironment.yml` – Our exported Conda environment file

## Acknowledgment  

The function `computeOffset` imported in `deformable_model_builder.py` is **adapted** from the original implementation in [Depth-Adapted CNN (GitHub)](https://github.com/Zongwei97/Depth-Adapted-CNN/blob/main/aCNN.py) by **Wu et al. (ACCV 2020)**.  

We modified the function to ensure full CUDA compatibility by explicitly moving all tensor operations to the GPU. Aside from these adjustments for improved performance on our CUDA setup, the core algorithm and functionality remain unchanged.
For the original implementation, please refer to the official repository: [https://github.com/Zongwei97/Depth-Adapted-CNN](https://github.com/Zongwei97/Depth-Adapted-CNN).  

If you use this work, please **cite their paper** as follows:  
```bibtex
@inproceedings{wu2020depth,
  title={Depth-adapted CNN for RGB-D cameras},
  author={Wu, Zongwei and Allibert, Guillaume and Stolz, Christophe and Demonceaux, C{\'e}dric},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  year={2020}
}
```

## Important Notes  
- **This repository does not include datasets or data-loading scripts.**  
- **Requests for data and preprocessing scripts will be considered upon request.**  
- This code is provided **as-is** with no guarantees.

## License  
This code is released under ****[Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/)****.

```text
Copyright (c) 2025, Mihaela-Larisa Clement.
This code is provided for academic reference. For any other use, please contact the authors.
```
