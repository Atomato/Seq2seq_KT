# DKT+

 - DKT 모델의 단점을 보완한 [DKT+](https://github.com/ckyeungac/deep-knowledge-tracing-plus) 모델 기반 기능 추가 개발.
 
### Dataset Description

  - [Individual BKT](http://gitlab.tmaxwork.shop/hyperstudy/knowledgetracing/python_kt_unitknowledgetracing/-/tree/individual_bkt)의 individual model dataset을 활용.
  - 이를 DKT+ dataset format으로 변환.
    - 첫 번째 row = 문제 풀이 시퀀스 개수
    - 두 번째 row = 스킬 id 시퀀스 (기존 데이터의 skill id를 0 ~ max_skill_num-1 로 맵핑)
    - 세 번재 row = 정오답 시퀀스


  ```
  15
  1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
  0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
  ```

### Model Addition

1. **Skill Embedded DKT+**
    - Input과 LSTM input 사이 embedding layer 추가 (embedding 차원=200)
    - main.py의 network_config['embedding'] = *True*/*False* 로 조절


### How To Run

```
pip install -r requirements.txt
python main.py
```

### Test Result

#### 1. Assistment 2009
  - 5번의 모델 학습 반복 및 평균 성능 기록.
  - ./results/a2009u/ 에 결과 저장.

1. **model = DKT**
    - $\lambda_0$ = 0.0, $\lambda_{w1}$ = 0.0, $\lambda_{w2}$ = 0.0

    - embedding = *False*

      ```
      The best testing result occured at: 6-th epoch, with testing AUC: 0.82165
      *********************************
      average AUC for 5 runs: 0.8220155373849579
      average AUC Current for 5 runs: 0.870065672603247
      average waviness-l1 for 5 runs: 0.07280272439391001
      average waviness-l2 for 5 runs: 0.11449000715743937
      average consistency_m1 for 5 runs: 0.2798314292562115
      average consistency_m1 for 5 runs: 0.0033070810666559146
      ```

    - embedding = *True*

      ```
      The best testing result occured at: 11-th epoch, with testing AUC: 0.82561
      *********************************
      average AUC for 5 runs: 0.8259260836995208
      average AUC Current for 5 runs: 0.8538206911834502
      average waviness-l1 for 5 runs: 0.10955865416504809
      average waviness-l2 for 5 runs: 0.17086800366663718
      average consistency_m1 for 5 runs: 0.24320923543980189
      average consistency_m1 for 5 runs: -0.0015450966547946347
      ```

2. **model = DKT+**
    - $\lambda_0$ = 0.1, $\lambda_{w1}$ = 0.03, $\lambda_{w2}$ = 3.0

    - embedding = *False*

      ```
      The best testing result occured at: 15-th epoch, with testing AUC: 0.82504
      *********************************
      average AUC for 5 runs: 0.824335969028188
      average AUC Current for 5 runs: 0.9552285615588921
      average waviness-l1 for 5 runs: 0.02060184455219459
      average waviness-l2 for 5 runs: 0.04679744589296681
      average consistency_m1 for 5 runs: 0.40512902452664373
      average consistency_m1 for 5 runs: 0.06847989052936485
      ```

    - embedding = *True*

        ```
        The best testing result occured at: 15-th epoch, with testing AUC: 0.82623
        *********************************
        average AUC for 5 runs: 0.8258746705334745
        average AUC Current for 5 runs: 0.9540060738507712
        average waviness-l1 for 5 runs: 0.021970434304943567
        average waviness-l2 for 5 runs: 0.04869209260507832
        average consistency_m1 for 5 runs: 0.40452983941839105
        average consistency_m1 for 5 runs: 0.06714509249541853
        ```

### ToDo

1. **User specific DKT+**
    - user_id에 따라 initial hidden state 값을 달리 설정.
    - user embedding 학습.
    - independent DKT.
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


### Detail hyperparameter for the program
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


### Citation

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