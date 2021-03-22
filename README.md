
# ROAR
Remove and Retrain [[1]](#1), ROAR for short, is a benchmark for evaluating interpretability methods in deep neural networks.
This repository provides implementation of ROAR metric which was used in our work [[2]](#2). 

## Introduction

ROAR leverages the fact that if an attribution method ranks pixels, then removing most important pixels from the dataset will lead
to worse training accuracy. Thus the better the attribution method, the worse will be model trained on this altered dataset.

ROAR consists of 3 steps:-

1. Train a classification model on original dataset.
2. Use the trained model to extract attribution maps from original dataset.
3. Retrain the classification model on altered dataset, where this dataset is created by removing top K% most important pixels from original dataset.

## Implementation

Note - The original author provided the source code for their implementation [here](https://github.com/google-research/google-research/tree/master/interpretability_benchmark).
Their implementation utilizes Tensorflow, TPU and many simplistic attribution methods which can be computed during training. This causes few issues:-

<ol type="a">
  <li>TPU is not available to all.</li>
  <li>Computing attribution methods during training can cause a very heavy bottleneck if attribution method takes too long to compute. A prime example for this is Integrad which reuires 50-100 forward passes per image.</li>
</ol>

### Setup

Requirements: Repository uses Taskfile(optional) and docker. You also need nvidia driver setup to use gpu for model training.

Install taskfile from [here](#https://taskfile.dev/#/installation). 

```shell
git clone https://github.com/saurabheights/ROAR.git
cd ROAR
task build-docker

# Ssh to docker
task bash-docker

# Train model (by default uses Resnet8 with CIFAR10)
task train-cls

# Extract attribution maps
task extract_attribution_maps

# Retrain on 
task evaluate_attribution_maps_roar
```


### Adding new dataset

WIP - BirdSnap and Food101 will be added soon. If you are interested in your own dataset, follow below instructions.

ToDo - Add instructions.

### Adding a new attribution method

By default, pipeline supports multiple attribution methods. Add your methods to config file as:- 

```yaml
extract_cams:
    attribution_methods: &ATTRIBUTION_METHODS
        - name: gradcam
          method: src.attribution_loader.compute_gradcam
          kwargs:
              saliency_layer: 'layer3.0.relu'
        - name: new_method
          method: method_to_call
          kwargs:
            any_custom_args_to_pass: new_arg_values 
```

Each attribution method should accept following parameters `model, preprocessed_image, label` and 
return an attribution map of same size as Image, representing importance of each input pixel. To pass 
any extra information, use `extract_cams.attribution_methods[index].kwargs` in yaml configuration file.

You can check `src.attribution_methods.attribution_loader.generate_attribution` to see how each attribution method is called.

## Contributions

For any questions, bugs or feature enhancement, simply make an [issue](https://github.com/saurabheights/ROAR/issues).

## Citation

If you use this work, please do cite both references.

<a id="1">[1]</a> 
Hooker, Sara, et al. "A Benchmark for Interpretability Methods in Deep Neural Networks." NeurIPS. 2019.

<a id="2">[2]</a> 
ToDo - Add our citation.
