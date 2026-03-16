
# HALF

PyTorch Code for the following paper:

Title: HALF: A Homophily-Aware Loss Fusion for Robust Learning under Label Noise in Heterophilic Graphs

Keywords: graph neural networks, semi-supervised node classification, label noise, heterophilic graphs


### Abstract

The message-passing mechanism in Graph Neural Networks (GNNs) amplifies the negative effects of label noise. This undermines the robustness of GNNs.
While recent methods aim to mitigate this issue, most rely on the homophily assumption, limiting their applicability to heterophilic graphs. 
In this paper, we conduct an empirical study on the ability of low- and high-pass losses to distinguish clean from noisy labeled nodes on both homophilic and heterophilic graphs.
Motivated by our findings, we propose Homophily-Aware Loss Fusion (HALF), a novel method for robust learning under label noise.
The key idea of HALF is to adaptively identify clean labeled nodes by combining low- and high-pass losses.
Specifically, we first introduce a dual-structure constructor to generate homophilic and heterophilic graph structures from node features.
Then, we propose a loss fusion that adaptively combines low- and high-pass losses, guided by node-level local homophily.
Finally, clean and noisy labeled nodes are identified using the memorization effect based on the fused loss.
To achieve more effective supervision for unlabeled nodes, we further propose a dual-pass alignment loss that aligns predictions from the low- and high-pass filters.
We conduct extensive experiments across diverse homophilic and heterophilic graphs under label noise. The results demonstrate the outperformance and adaptability of HALF over state-of-the-art baselines.


### Requirements
- python >= 3.10
- numpy >= 2.2.6
- pytorch >= 2.5.0
- torch_geometric >= 2.6.1


### Running NormProp

We have pre-generated some prototypes and saved them. You can run HALF directly using the following command.
```bash
# citeseer
python single_exp.py --method adaptloss --data citeseer --noise_type uniform --noise_rate 0.2 --device cuda:1 --seed 3000 
python single_exp.py --method adaptloss --data citeseer --noise_type uniform --noise_rate 0 --device cuda:0 --seed 3000  


# flickr
python single_exp.py --method adaptloss --data flickr --noise_type uniform --noise_rate 0 --device cuda:1 --seed 3000  

```


### Hyperparameter optimization.
``` bash

python hyperparam_opt.py --method adaptloss --data citeseer --noise_type uniform --noise_rate 0.2 --device cuda:1 --max_trial_number 200 --trial_concurrency 1 --update_config True  --port 8091

python hyperparam_opt.py --method adaptloss --data flickr --noise_type uniform --noise_rate 0.2 --device cuda:1 --max_trial_number 200 --trial_concurrency 2 --update_config True  --port 8092
```

By running the command above, an NNI manager will run on http://localhost:8081, 
then automatically run 20 HPO trails, each trail call 'single_exp.py' with different hyperparameters. 












