## Cropland-CD

The pytorch implementation for **MSCANet** in paper "[A CNN-transformer Network with Multi-scale Context Aggregation for Fine-grained Cropland Change Detection](https://ieeexplore.ieee.org/document/9780164)" on [IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing](https://www.grss-ieee.org/publications/journal-of-selected-topics-in-applied-earth-observations-and-remote-sensing/).  

## Requirements
- Python 3.6
- Pytorch 1.7.0


## Datasets
### CropLand Change Dection (CLCD) Dataset
The CLCD dataset consists of 600 pairs image of cropland change samples, with 360 pairs for training, 120 pairs for validation and 120 pairs for testing.
The bi-temporal images in CLCD were collected by Gaofen-2 in Guangdong Province, China, in 2017 and 2019, respectively, with spatial resolution ranged from 0.5 to 2 m. Each group of samples is composed of two images of 512 × 512 and a corresponding binary label of cropland change.

- Download the CLCD Dataset: [OneDrive](https://mail2sysueducn-my.sharepoint.com/:f:/g/personal/liumx23_mail2_sysu_edu_cn/Ejm7aufQREdIhYf5yxSZDIkBr68p2AUQf_7BAEq4vmV0pg?e=ZWI3oy) | [Baidu](https://pan.baidu.com/s/1Un-bVxUm1N9IHiDOXLLHlg?pwd=miu2)
- Download the [HRSCD Dataset](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset)





## Citation

Please cite our paper if you use this code in your work:

```
@ARTICLE{9780164,
  author={Liu, Mengxi and Chai, Zhuoqun and Deng, Haojun and Liu, Rong},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A CNN-Transformer Network With Multiscale Context Aggregation for Fine-Grained Cropland Change Detection}, 
  year={2022},
  volume={15},
  number={},
  pages={4297-4306},
  doi={10.1109/JSTARS.2022.3177235}}
```
