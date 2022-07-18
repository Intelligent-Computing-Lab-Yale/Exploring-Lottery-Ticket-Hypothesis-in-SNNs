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

## Implementation 

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
