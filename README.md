# DKT+

 - DKT 모델의 단점을 보완한 [DKT+](https://github.com/ckyeungac/deep-knowledge-tracing-plus) 모델을 assistment 2009 데이터에 적용.
 
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

  - DKT+skill embedding layer
    - Input과 LSTM input 사이 embedding layer 추가 (embedding 차원=200)
    - main.py의 network_config['embedding'] = True/False 로 조절


### How To Run

```
pip install -r requirements.txt
python main.py
```

### Test Result

 - ./assist_lsh/ 에 결과 저장

 ```
The best testing result occured at: 1-th epoch, with testing AUC: 0.72354
*********************************
average AUC for 1 runs: 0.7235410191831241
average AUC Current for 1 runs: 0.8546168541225564
average waviness-l1 for 1 runs: 0.0837035549925903
average waviness-l2 for 1 runs: 0.1313468509359241
average consistency_m1 for 1 runs: 0.4370631960117092
average consistency_m1 for 1 runs: 0.06105880186848471

program run for: 5484.073899269104s
 ```


### ToDo

  - DKT + attention : 지추추 결과, 특정 UK에 대한 지식 수준이 낮게 나온 원인이 되는 문제를 도출 --> 맞춤형 강의
  - DKT + user_id : user_id를 initial hidden state 값으로 설정 --> independent DKT

  - CNN 모델을 통해 해석가능한 AI 여부 조사 예정
    - CNN의 feature map을 활용.
    - sequential data에 대한 CNN 적용 사례를 기반으로, feature map 강도에 따라 학습 정도를 파악할 수 있음.
    - 모델이 잘 학습할 수 있는 input 시퀀스, 즉 문제 시퀀스를 생성.


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