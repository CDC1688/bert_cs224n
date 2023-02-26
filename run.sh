#!/usr/bin/env bash

NUM_EPOCHS=50
LR=1e-5
BATCH_SIZE=256
RATE=0.3


python classifier.py --option pretrain –-epochs$ NUM_EPOCHS –-lr $LR –-batch_size $BATCH_SIZE --hidden_dropout_prob $RATE