## Dataset
- Unzip data.zip to `./{current_working_directory}`
    ```sh
    unzip data.zip -d ./{current_working_directory}
    ```
- Manually split the first 100 jpgs of each label in "test" direstory to the other directory named "test_finetune".
    - Remember to keep folder structure identical to the "test" directory.
- Folder structure
    ```
    .
    ├── unlabeled
    │   └── nolabel/  
    ├── test
    │   ├── 0/ 
    │   ├── 1/
    │   ├── 2/
    │   └── 3/ 
    ├── test_finetune
    │   ├── 0/ 
    │   ├── 1/
    │   ├── 2/
    │   └── 3/ 
    ├── model_weight
    ├── 310704014.npy
    ├── create_embed.py
    ├── main.py
    ├── prepare_data_path.py
    ├── prepare_fine_tune_data_path.py
    ├── prepare_test_data_path.py
    ├── Readme.md
    ├── requirements.txt
    ├── leave-one-out-CV_KNN.py
    ├── data.csv
    ├── data_finetune.csv
    └── data_test.csv
    ```

## Environment
- Python 3.9 or later version
    ```sh
    conda create --name <env> --file requirements.txt
    ```

## Preprocess unlabeled Pre-training data
- Saving the jpgs paths.
- Create a csv file named as `data.csv`
```sh
python prepare_data_path.py
```

## Preprocess labeled test_finetune data
- Saving the jpgs paths.
- Create a csv file named as `data_finetune.csv`
```sh
python prepare_fine_tune_data_path.py
```

## Preprocess labeled test data
- Saving the jpgs paths.
- Create a csv file named as `data_test.csv`
```sh
python prepare_test_data_path.py
```

## Train & fine-tuning & testing
- ResNet18
- With RTX 2080ti and 128GB RAM, it may cost 2 hours to train.
- The pre-trained model weight will be saved in folder "model_weight", its weight name **may be** `SimCLR_RN18_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_150_20221031.pt`.
- The fine-tuned model weight will be saved in folder "model_weight", its weight name **may be** `rn18_p128_sgd0p01_decay0p98_all_lincls_20221031.pt`.
- The testing accuracy would be displayed.
```sh
python main.py
```

## Get the embedding of unlabeled data
- Please use the correct model name and its corresponding weight.
- The output would be the embedding representation of the pretrained feature extractor ResNet18.
```sh
python create_embed.py
```
The numpy file **may be** `310704014.npy`. Named it anything you want.