# Skill Embedded DKT

- DKT 모델의 단점을 보완한 [DKT+](https://github.com/ckyeungac/deep-knowledge-tracing-plus) 모델 기반 지식 추적 모델 구현.
 
**Skill Embedded DKT+**

- Parameter

```
- network_config['emb_layer'] = True/False
- network_config['embedding_dims'] = 200
- network_config['skill_separate_emb'] = True/False
- network_config['expand_correct_dim'] = True/False
```

- *emb_layer* : data input과 LSTM input 사이 embedding layer 추가.
- *skill_separate_emb* : skill의 embedding과 correct 정보를 분리.
- *expand_correct_dim* : correct 정보의 효과를 위한 차원 확장.

 
 
## Dataset Description

- [Individual BKT](http://gitlab.tmaxwork.shop/hyperstudy/knowledgetracing/python_kt_unitknowledgetracing/-/tree/individual_bkt)의 [ASSISTment_skill_builder_only_*_1123.txt](http://gitlab.tmaxwork.shop/hyperstudy/knowledgetracing/python_kt_unitknowledgetracing/-/blob/individual_bkt/data/ASSISTment_skill_builder_only_train_1123.txt) 데이터 활용.

- ./data/ASSISTment_skill_builder_only_1127/assitment_1127_*.csv

### Data Format
  - 첫 번째 row = user id (0 ~ num_user-1)
  - 두 번째 row = skill id sequence (0 ~ num_skill-1)
  - 세 번재 row = correct sequence (0/1)

  ```
  308
  1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
  0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
  ```

### Data 통계
  - 총 skill 개수 (*num_skill*) = 105
  - 총 user 수 (*num_user*) = 4003
  - train_data : \# = 3661, *max_length* = 511
  - valid data : \# = 1978, *max_length* = 127
  - test data : \# = 2895, *max_length* = 213


## How To Run

```
pip install -r requirements.txt
(main.py 내 파라미터 조절 후) python main.py
```


## Test Result

### 1. ASSISTment_skill_builder_only_1127

  - 5번의 모델 학습 반복 및 평균 성능 기록.
  - ./results/assistment_1127/logs/ 에 결과 저장.


**1. DKT**

- $\lambda_0$ = 0, $\lambda_{w1}$ = 0, $\lambda_{w2}$ = 0

  ```
  average validation ACC for 5 runs: 0.722472097284691
  average validation AUC for 5 runs: 0.7312790003966417
  average validation AUC Current for 5 runs: 0.880834532582383
  ...

  test ACC for 5 runs : 0.7251, 0.7228, 0.7289, 0.7246, 0.7232
  test AUC for 5 runs : 0.7523, 0.7509, 0.7563, 0.7545, 0.7524

  average test ACC for 5 runs: 0.7249339804439369
  average test AUC for 5 runs: 0.7532964399989337
  average test AUC Current for 5 runs: 0.8880471219012704
  ```


**2. DKT+**

- $\lambda_0$ = 0.1, $\lambda_{w1}$ = 0.003, $\lambda_{w2}$ = 3.0

**(1) emb_layer = *False***

  - 기존 DKT+ 모델과 동일.

    ```
    average validation ACC for 5 runs: 0.725144094619357
    average validation AUC for 5 runs: 0.734958509238247
    average validation AUC Current for 5 runs: 0.9364238904704576
    ...

    test ACC for 5 runs : 0.73044, 0.73135, 0.73068, 0.73101, 0.72991
    test AUC for 5 runs : 0.7606, 0.76306, 0.76211, 0.76395, 0.76063

    average test ACC for 5 runs: 0.7306794661337521
    average test AUC for 5 runs: 0.7620699804997882
    average test AUC Current for 5 runs: 0.9384023639959386
    ```


**(2)  emb_layer = *True***

  - skill_separate_emb = *False*

    ```
    average validation ACC for 5 runs: 0.7271697484591038
    average validation AUC for 5 runs: 0.7369011724812694
    average validation AUC Current for 5 runs: 0.931571503793674
    ...
    
    test ACC for 5 runs : 0.73143, 0.7323, 0.731, 0.73035, 0.73291
    test AUC for 5 runs : 0.7612, 0.76448, 0.762, 0.76177, 0.76169

    average test ACC for 5 runs: 0.7315966026693312
    average test AUC for 5 runs: 0.7622270136916077
    average test AUC Current for 5 runs: 0.9371024762776695
    ```

  - **skill_separate_emb = *True* & expand_correct_dim = *False* \***

    ```
    average validation ACC for 5 runs: 0.7258170914542729
    average validation AUC for 5 runs: 0.7364152991930496
    average validation AUC Current for 5 runs: 0.9143275528897513
    ...
    
    test ACC for 5 runs : 0.7316, 0.73262, 0.73285, 0.73168, 0.73362
    test AUC for 5 runs : 0.76479, 0.76267, 0.76594, 0.76365, 0.76206

    average test ACC for 5 runs: 0.7324744843337377
    average test AUC for 5 runs: 0.7638218787225003
    average test AUC Current for 5 runs: 0.9199936315038842
    ```

  - skill_separate_emb = *True* & expand_correct_dim = *True*

    ```
    average validation ACC for 5 runs: 0.7229385307346327
    average validation AUC for 5 runs: 0.7305807564287312
    average validation AUC Current for 5 runs: 0.9280161092685832
    ...

    test ACC for 5 runs : 0.73057, 0.73125, 0.73041, 0.72877, 0.73066
    test AUC for 5 runs : 0.75755, 0.7586, 0.75912, 0.75871, 0.75999

    average test ACC for 5 runs: 0.7303297409178503
    average test AUC for 5 runs: 0.7587946868933916
    average test AUC Current for 5 runs: 0.9342358026920173
    ```

## ToDo

1. **User specific DKT+**
    - user_id에 따라 initial hidden state 값을 달리 설정. [user_specific_dkt](http://gitlab.tmaxwork.shop/hyperstudy/knowledgetracing/python_kt_unitknowledgetracing/-/tree/user_specific_dkt)
    - user embedding 학습.
<br>

2. **Attention DKT+**
    - Embedding layer output에 대한 attention 적용.
    - Hidden state들에 대한 attention 적용.
    - 특정 UK에 대한 지식 수준이 낮게 나온 원인이 되는 문제를 도출.
    - 맞춤형 강의에 활용 가능.
<br>

3. **Convolutional DKT**
    - CNN의 feature map을 활용.
    - sequential data에 대한 CNN 적용 사례를 기반으로, feature map 강도에 따라 모델의 학습 정도를 파악할 수 있음.
    - 모델이 잘 학습할 수 있는 input 시퀀스, 즉 문제 시퀀스를 생성 및 추천


## Detail hyperparameter for the program

- 기존 DKT+ repository

```
usage: main_origin.py [-h]
               [-hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]]]
               [-cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}]
               [-lr LEARNING_RATE] [-kp KEEP_PROB] [-mgn MAX_GRAD_NORM]
               [-lw1 LAMBDA_W1] [-lw2 LAMBDA_W2] [-lo LAMBDA_O]
               [--num_runs NUM_RUNS] [--num_epochs NUM_EPOCHS]
               [--batch_size BATCH_SIZE] [--data_dir DATA_DIR]
               [--train_file TRAIN_FILE] [--test_file TEST_FILE]
               [-csd CKPT_SAVE_DIR] [--dataset DATASET]

optional arguments:
  -h, --help            show this help message and exit
  -hl [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]], --hidden_layer_structure [HIDDEN_LAYER_STRUCTURE [HIDDEN_LAYER_STRUCTURE ...]]
                        The hidden layer structure in the RNN. If there is 2
                        hidden layers with first layer of 200 and second layer
                        of 50. Type in '-hl 200 50'
  -cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}, --rnn_cell {LSTM,GRU,BasicRNN,LayerNormBasicLSTM}
                        Specify the rnn cell used in the graph.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate when training the model.
  -kp KEEP_PROB, --keep_prob KEEP_PROB
                        Keep probability when training the network.
  -mgn MAX_GRAD_NORM, --max_grad_norm MAX_GRAD_NORM
                        The maximum gradient norm allowed when clipping.
  -lw1 LAMBDA_W1, --lambda_w1 LAMBDA_W1
                        The lambda coefficient for the regularization waviness
                        with l1-norm.
  -lw2 LAMBDA_W2, --lambda_w2 LAMBDA_W2
                        The lambda coefficient for the regularization waviness
                        with l2-norm.
  -lo LAMBDA_O, --lambda_o LAMBDA_O
                        The lambda coefficient for the regularization
                        objective.
  --num_runs NUM_RUNS   Number of runs to repeat the experiment.
  --num_epochs NUM_EPOCHS
                        Maximum number of epochs to train the network.
  --batch_size BATCH_SIZE
                        The mini-batch size used when training the network.
  --data_dir DATA_DIR   the data directory, default as './data/
  --train_file TRAIN_FILE
                        train data file, default as 'skill_id_train.csv'.
  --test_file TEST_FILE
                        train data file, default as 'skill_id_test.csv'.
  -csd CKPT_SAVE_DIR, --ckpt_save_dir CKPT_SAVE_DIR
                        checkpoint save directory
  --dataset DATASET
  -emb EMBEDDING, --embedding EMBEDDING
                        Whether to add skill embedding layer after input
```


## Citation

This is the repository for the code in the paper *Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization* ([ACM](https://dl.acm.org/citation.cfm?id=3231647), [pdf](https://arxiv.org/pdf/1806.02180.pdf))

```
@inproceedings{LS2018_Yeung_DKTP,
  title={Addressing two problems in deep knowledge tracing via prediction-consistent regularization},
  author={Yeung, Chun Kit and Yeung, Dit Yan},
  year={2018},
  booktitle = {{Proceedings of the 5th ACM Conference on Learning @ Scale}},
  pages = {5:1--5:10},
  publisher = {ACM},
}
```