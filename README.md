# Understanding-Rich-Temporal-Information-for-TAL

### Recommended Environment
* Python 3.6+
* Pytorch 1.6+
* CUDA 10.2+
* scipy
* pandas
* joblib
* tqdm
* Tensorboard-logger
* Tensorboard

### Install
You can set up the environments by using `$ pip3 install -r requirements.txt`.

If there are any issues with installing PyTorch, run the command below following this [repo](https://github.com/workjo/Learning-Action-Completeness-from-Points/blob/main/Troubleshooting_the_problem_that_does_not_support_RTX_3090.md)
~~~~
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
~~~~

### Data Preparation
1. Download extracted features (THUMOS'14) from [LACP](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points).

2. Place the features inside the `dataset` folder.
    - Please ensure the data structure is as below.
   
~~~~
├── dataset
   └── THUMOS14
       ├── gt.json
       ├── split_train.txt
       ├── split_test.txt
       ├── fps_dict.json
       ├── point_gaussian
           └── point_labels.csv
       └── features
           ├── train
               ├── rgb
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
               └── flow
                   ├── video_validation_0000051.npy
                   ├── video_validation_0000052.npy
                   └── ...
           └── test
               ├── rgb
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
               └── flow
                   ├── video_test_0000004.npy
                   ├── video_test_0000006.npy
                   └── ...
~~~~

## Usage

### Training
By executing the script provided below, you can easily train the model.

If you want to try other options, please refer to 'options.py'.

~~~~
python main.py
~~~~

### Testing
The pre-trained model can be found [here](https://drive.google.com/file/d/1Tu2fTaXfAvjMhwWoy02_1z2qoEozCkfn/view?usp=sharing).
You can test the model by running the command below.
* Place the pre-trained model in the 'models' directoty.

~~~~
python main_eval.py
~~~~

## References
We referenced the repo below for the code.
Learning Action Completeness from Points for Weakly-supervised Temporal Action Localization (ICCV 2021) [[paper](https://arxiv.org/abs/2108.05029)] [[code](https://github.com/Pilhyeon/Learning-Action-Completeness-from-Points)]
