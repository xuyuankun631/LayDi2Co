# <p style="text-align: center;">LayDi2Co</p>

This is the code for our paper "LayDi2Co: A Discrete-to-Continuous Diffusion Model for Structured Layout Generation"




### Dependencies
- Operating System: Ubuntu 18.04
- CUDA Version: 11.6
- Python3
- Python Version: 3.9
### Requirements
All relevant requirements are listed in [environment.yml](environment.yml). 

### Training
You can train the model using any config script in [configs](./LayDi2Co/configs) folder. For example, if you want to train the 
provided LayDi2Co model on publaynet dataset, the command is as follows:

```bash
cd LayDi2Co
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/publaynet_config.py --workdir <WORKDIR>
```
Please, see that code is accelerator agnostic. if you don't want to log results to wandb, just set `--workdir test` 
in args.

### Evaluation

To generate samples for evaluation on the test set, follow these steps:

- train the model using the above command
- Run the following command:

```bash
# put weights in config.logs folder
DATASET = "publaynet" # or "rico"
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --config configs/{$DATASET}_config.py \\
                           --workdir <WORKDIR> --epoch <EPOCH> --cond_type <COND_TYPE> \\
                           --save True
# get all the metrics 
# update path to pickle file in LayDi2Co/evaluation/metric_comp.py
./download_fid_model.sh
python metric_comp.py
```
where `<COND_TYPE>` can be: (all, whole_box, loc) - (unconditional, category, category+size) respectively,
`<EPOCH>` is the epoch number of the model you want to evaluate, and `<WORKDIR>` is the path to the folder where
the model weights are saved (e.g. rico_final). The generated samples will be saved in `logs/<WORKDIR>/samples` folder if `save` True.

An output from it is pickle file with generated samples. You can use it to calculate metrics.

The folder with weights after training has this structure:
```
logs
├── publaynet_final
│   ├── checkpoints
│   └── samples
└── rico_final
    ├── checkpoints
    └── samples
```
### Datasets
Please download the public datasets at the following webpages. Put it in your folder and update 
`./LayDi2Co/configs/dataset_config.py` accordingly.

1. [RICO](https://interactionmining.org/rico)
2. [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)

### Example Results


- Comparison of quantitative results (i.e., pIOU, Overlap, Alignment and FID) of layout generation on Rico with various constraint settings. ↓: Lower is better; - : no results are reported. The values of Alignment, Overlap, and pIOU are multiplied by 100 for clarity, with the best and second-best results highlighted in bolded and underlined, respectively. Our method provides general superior results over the metrics.
![](Figure/image1.png)
![](Figure/image2.png)

- Qualitative comparison of layout generation results on the PubLayNet dataset under three conditioning settings: category + size, category only, and unconditioned generation. 
The first column shows the ground-truth layouts, while the last column presents the results produced by our method.
![](Figure/image3.png)

- We obtain four samples from model to demonstrate the effect of model generation
![](Figure/image4.png)






