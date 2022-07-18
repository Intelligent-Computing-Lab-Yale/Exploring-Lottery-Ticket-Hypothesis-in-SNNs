# Exploring-Lottery-Ticket-Hypothesis-in-Sparse-SNNs
Exploring Lottery Ticket Hypothesis in Sparse Spiking Neural Networks (ECCV2022, oral presentation)




## Environment
* Python 3.9    
* PyTorch 1.10.0   
* Spikingjelly
```
git clone https://github.com/fangwei123456/spikingjelly.git
cd spikingjelly
python setup.py install
```       

## Introduction 
Spiking Neural Networks (SNNs) have recently emerged as a new generation of low-power deep neural networks, which is suitable to be implemented on low-power mobile/edge devices. As such devices have limited memory storage, neural pruning on SNNs has been widely explored in recent decades. Most existing SNN pruning works focus on shallow SNNs (2~6 layers), however, deeper SNNs (>16 layers) are proposed by state-of-the-art SNN works, which is difficult to be compatible with the current SNN pruning work. To scale up a pruning technique towards deep SNNs, we investigate Lottery Ticket Hypothesis (LTH) which states that dense networks contain smaller subnetworks (i.e., winning tickets) that achieve comparable performance to the dense networks. Our studies on LTH reveal that the winning tickets consistently exist in deep SNNs across various datasets and architectures, providing up to 97% sparsity without huge performance  degradation. However, the iterative searching process of LTH brings a huge training computational cost when combined with the multiple timesteps of SNNs. To alleviate such heavy searching cost, we propose Early-Time (ET) ticket where we find the important weight connectivity from a smaller number of timesteps. The proposed ET ticket can be seamlessly combined with a common pruning techniques for finding winning tickets,such as Iterative Magnitude Pruning (IMP) and Early-Bird (EB) tickets. 

<img width="784" alt="Screen Shot 2022-07-18 at 10 41 11 AM" src="https://user-images.githubusercontent.com/41351363/179536579-8de254ba-5fe5-47c1-86fd-c570bf746069.png">



## Implementation 

<img width="998" alt="Screen Shot 2022-07-18 at 10 39 46 AM" src="https://user-images.githubusercontent.com/41351363/179536311-714697ab-7355-42ae-942e-bbf8216a4e0e.png">

›
1) Iterative Magnitude Pruning (IMP)
```
python train_snn_laterewindlth.py  --dataset 'cifar10' --arch 'vgg16' --optimizer 'sgd'  --batch_size 128 --learning_rate 3e-1 
```
2) Early-Bird ticket (EB)
```
python train_snn_earlybird.py  --dataset 'cifar10' --arch 'vgg16' --optimizer 'sgd' --sparsity_round 4  --batch_size 128 --learning_rate 3e-1  --round 1
```
3) IMP + Early-Time ticket (ET)
```
python train_snn_earlyT_laterewindlth.py  --dataset 'cifar10' --arch 'vgg16' --optimizer 'sgd'  --batch_size 128 --learning_rate 3e-1 
```
4) EB + Early-Time ticket (ET)
```
python train_snn_earlyT_earlybird.py  --dataset 'cifar10' --arch 'vgg16' --optimizer 'sgd' --sparsity_round 4  --batch_size 128 --learning_rate 3e-1  --round 1
```
*  --sparsity_round: select sparsity level in sparsity list = [0,	25.04,	43.71, 57.76,	68.3, 76.2,	82.13, 86.58, 89.91, 92.41,	94.29, 95.69, 96.75, 97.54,	98.13]


## Acknowledgement 
The overall IMP and LTH codes are referred from: 
https://github.com/rahulvigneswaran/Lottery-Ticket-Hypothesis-in-Pytorch

ET code is borrowed from: 
https://github.com/RICE-EIC/Early-Bird-Tickets

## Citation
```
@article{kim2022lottery,
  title={Lottery Ticket Hypothesis for Spiking Neural Networks},
  author={Kim, Youngeun and Li, Yuhang and Park, Hyoungseob and Venkatesha, Yeshwanth and Yin, Ruokai and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2207.01382},
  year={2022}
}
```       
