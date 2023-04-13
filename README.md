# Semi-supervised Edge-aware Point Cloud Upsampling Network
An implementation of 3D Deep Learning and Traditional Computer Vision techniques to accurately upsample point clouds while being edge aware and respecting finer details. 

## Proposed Architecture
![Architecture Edge-Punet](https://drive.google.com/file/d/16CdE2YnmeS-TXeWw9Ehmnosr_6Od64HJ/view?usp=share_link "Architecture")

## Requirements
1. Python v3.9
2. Pytorch v1.13.0
3. CUDA v11.7
4. Pytorch3d (version that is compatible with the above requirements. Follow [this](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) for installation)
5. (Optional) COLMAP (if you wish to use multiview images as input for training/testing)

## Dataset
Stanford Large-Scale 3D Indoor Spaces Dataset (S3DIS) was used for training and testing this model. However this dataset was slightly modified and the best results were obtained when training on a custom made patches of scenes of the environment. 

#### Citation for S3DIS
```
@InProceedings{armeni_cvpr16,  
  title ={3D Semantic Parsing of Large-Scale Indoor Spaces},  
  author = {Iro Armeni and Ozan Sener and Amir R. Zamir and Helen Jiang and Ioannis Brilakis and Martin Fischer and Silvio Savarese},  
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision and Pattern Recognition},  
  year = {2016}, }
```

### Download S3DIS
1. Go to [Stanford S3DIS homepage](http://buildingparser.stanford.edu/dataset.html)
2. Scroll down to "Download" and click "here" on the S3DIS point. 
3. Fill the form and accept required terms and download the dataset.
4. Download the Aligned Version .zip

### Unpack S3DIS
1. Create a `data/s3dis/` directory in the root of this repository.
2. Unpack the .zip in this directory.
3. From the root, run this command.
```
cd data_utils
python collect_indoor3d_data.py
```
4. You should see the extracted data in `data/pointclouds/s3dis'`

### Custom S3DIS Dataset Download 
***(Note: Make sure to agree to the S3DIS terms and conditions before this!)***
1. Download the .zip from [here](https://drive.google.com/file/d/1EDrwDC0VqSEjMcT0DlIifqej7J16L37Q/view?usp=share_link)
2. Extract the repository inside `data/pointclouds/`

## Training
For training with the S3DIS dataset of object patches, run the below script from root directory.
```
python train.py 
  --model edgepunet
  --epochs 100
  --batchsize 8
  --npoint 1024
  --upsample_rate 4
  --counter default
```
Checkpoints are stored in `checkpoints/` directory. `checkpoints/[YourModel]/instant` contains checkpoints for inference whereas `checkpoints/[YourModel]` contains checkpoints for resuming training.

## Testing
Checkpoint for 100th epoch can be downloaded [here](https://drive.google.com/file/d/1wuBDPA95-4FGMvxnXhNfK0vt51XpfVOt/view?usp=share_link)

For Testing with Object Patches, make sure there a checkpoint to test on and run the below code in the root.
```
python train.py 
  --model edgepunet
  --checkpoint checkpoints/edgepunet_default/instant
  --epochs "[0, 50, 100]"
  --batchsize 1
  --npoint 1024
  --upsample_rate 4
```

## Results
Check for the results in the `out/train` for training and `out/test` for testing.

## Our Results with our Edge Aware Upsampling Model

### Training Results (shown for a single patch here)
![Training on Stair Patch](https://drive.google.com/file/d/14j5DUVkTzkdqgsOjDgmy48d3pG4VKgh9/view?usp=share_link "Training on Stair patch")

### Testing Results (shown for a single patch here)
![Training on Chair Patch](https://drive.google.com/file/d/1gUvGU1LX4jJhbpY5yUa42TPNx_DttqgB/view?usp=share_link "Testing on Chair Patch")

## References

#### PointNet
```
@article{qi2016pointnet,
  title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1612.00593},
  year={2016}
}
```

#### PointNet++
```
@article{qi2017pointnetplusplus,
  title={PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  author={Qi, Charles R and Yi, Li and Su, Hao and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1706.02413},
  year={2017}
}
```

#### The script for data extraction and Pointnet++ pytorch code used from [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) repository
```
@article{Pytorch_Pointnet_Pointnet2,
  Author = {Xu Yan},
  Title = {Pointnet/Pointnet++ Pytorch},
  Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  Year = {2019}
}
```

#### PU-Net
```
@inproceedings{yu2018pu,
  title={PU-Net: Point Cloud Upsampling Network},
  author={Yu, Lequan and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
  booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2018}
}
```

#### PU-Net Pytorch Code was used from [PU-Net_pytorch](https://github.com/lyqun/PU-Net_pytorch) repository

#### COLMAP 
```
@inproceedings{schoenberger2016sfm,
  author={Sch\"{o}nberger, Johannes Lutz and Frahm, Jan-Michael},
  title={Structure-from-Motion Revisited},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
}

@inproceedings{schoenberger2016mvs,
  author={Sch\"{o}nberger, Johannes Lutz and Zheng, Enliang and Pollefeys, Marc and Frahm, Jan-Michael},
  title={Pixelwise View Selection for Unstructured Multi-View Stereo},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2016},
}
```
