import time
import random
import numpy as np
import argparse
import sys
import re
import os
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data

from evaluation import model_eval_sst, test_model_multitask ,model_eval_multitask

#====add tensorbaord
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
TQDM_DISABLE = True

# fix the random seed


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        #self.shared=nn.Linear(config.hidden_size,512)
        self.setiment_layer=nn.Linear(config.hidden_size,128)
        self.activation1=torch.nn.SiLU()
        self.dropout_sen=nn.Dropout(0.3)
        self.output_sen=nn.Linear(128,5)
        self.para_pool1=torch.nn.AdaptiveAvgPool1d(64)
        self.para_pool2=torch.nn.AdaptiveMaxPool1d(64)
        self.para_drop=nn.Dropout(0.2)
        self.para_dense=nn.Linear(64*3,2)
        #self.lstm=nn.LSTM(input_size=config.hidden_size,hidden_size=128,num_layers=1,bias=True,bidirectional=True)
        self.cosine_pool1=torch.nn.AdaptiveAvgPool1d(64)
        self.cosine_pool2=torch.nn.AdaptiveMaxPool1d(64)
        self.cosine_drop=nn.Dropout(0.2)
        self.cosine_dense=nn.Linear(64*3,6)

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state=bert_output["last_hidden_state"]  #[8, 41, 768]
        pooler_output=bert_output["pooler_output"]    #[8, 768]

        output=last_hidden_state[:, 0]
        #output=torch.squeeze(torch.matmul(attention_mask.type(torch.float32).view(-1,1,attention_mask.shape[-1]),last_hidden_state),1)
        #output=self.shared(output)
        return output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        bert_output = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        sentiment_logits = self.setiment_layer(bert_output)
        sentiment_logits=self.activation1(sentiment_logits)
        sentiment_logits=self.dropout_sen(sentiment_logits)
        sentiment_logits=self.output_sen(sentiment_logits)
        return sentiment_logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        sent1_embeddings = self.forward(input_ids_1, attention_mask_1)
        sent2_embeddings = self.forward(input_ids_2, attention_mask_2)

        u=self.para_pool1(sent1_embeddings)   #[batch, 32]
        v=self.para_pool2(sent2_embeddings)   #[batch, 32]

        uv_diff=u-v
        out=torch.cat([u,v,uv_diff],dim=1)

        out=self.para_drop(out)
        out=self.para_dense(out)   #[batch, 6]
        #print(">>>>para logit size",out.shape)
        return out

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # get the embeddings for the first sentence
        sent1_embeddings = self.forward(input_ids=input_ids_1, attention_mask=attention_mask_1) #[batch, 768]
        sent2_embeddings = self.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
        #print(">>>embedding shape",sent2_embeddings.shape)    
        #lstm_encode, (last_hidden, last_cell)=self.lstm(embeddings)  #(src_len, b, h*2) for output
        #print(">>>lstm shape",bi_lstm.shape)
        u=self.cosine_pool1(sent1_embeddings)   #[batch, 32]
        v=self.cosine_pool2(sent2_embeddings)   #[batch, 32]
 
        uv_diff=u-v
        out=torch.cat([u,v,uv_diff],dim=1)

        out=self.cosine_drop(out)
        out=self.cosine_dense(out)   #[batch, 6]

        return out


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


# Currently only trains on sst dataset
def train_multitask(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_data.collate_fn)
    
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    #add tensorbaord
    tensorboard_path = args.run_path  
    writer = SummaryWriter(log_dir=f"./runs/{tensorboard_path}")

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(zip(sst_train_dataloader, para_train_dataloader,sts_train_dataloader),desc=f'train-{epoch}', disable=TQDM_DISABLE):
            sst,para,sts=batch
            # print(sst.keys())
            # print(sts.keys())
            # print(para.keys())
            sst_b_ids, sst_b_mask, sst_b_labels = (sst['token_ids'],
                                       sst['attention_mask'], sst['labels'])

            sst_b_ids = sst_b_ids.to(device)
            sst_b_mask = sst_b_mask.to(device)
            sst_b_labels = sst_b_labels.to(device)

            para_ids_1, para_mask_1, para_ids_2,para_mask_2,para_labels = (para['token_ids_1'],
                                       para['attention_mask_1'], para['token_ids_2'],para['attention_mask_2'],para['labels'])

            para_ids_1 = para_ids_1.to(device)
            para_mask_1 = para_mask_1.to(device)
            para_ids_2 = para_ids_2.to(device)
            para_mask_2 = para_mask_2.to(device)
            para_labels = para_labels.to(device)


            sts_ids_1, sts_mask_1, sts_ids_2,sts_mask_2,sts_labels = (sts['token_ids_1'],
                                       sts['attention_mask_1'], sts['token_ids_2'],sts['attention_mask_2'],sts['labels'])

            sts_ids_1 = sts_ids_1.to(device)
            sts_mask_1 = sts_mask_1.to(device)
            sts_ids_2 = sts_ids_2.to(device)
            sts_mask_2 = sts_mask_2.to(device)
            sts_labels = sts_labels.to(device)

            optimizer.zero_grad()
            logits_sst = model.predict_sentiment(sst_b_ids, sst_b_mask)
            loss_sst = F.cross_entropy(logits_sst, sst_b_labels.view(-1), reduction='sum') / args.batch_size

            logits_para = model.predict_paraphrase(para_ids_1, para_mask_1,para_ids_2,para_mask_2)
            loss_para = F.cross_entropy(logits_para, para_labels.view(-1), reduction='sum') / args.batch_size

            logits_sts = model.predict_similarity(sts_ids_1, sts_mask_1,sts_ids_2,sts_mask_2)
            loss_sts = F.cross_entropy(logits_sts, sts_labels.view(-1), reduction='sum') / args.batch_size
            
            loss=loss_sst+loss_para+loss_sts

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        
        paraphrase_accuracy, n1, n11,sentiment_accuracy,n2, n22,sts_accuracy, n3, n33=model_eval_multitask(sst_train_dataloader,
                                                para_train_dataloader,
                                                sts_train_dataloader,
                                                model, device)
        
        paraphrase_accuracy_dev, n12, n12,sentiment_accuracy_dev,n22, n222,sts_accuracy_dec, n33, n333=model_eval_multitask(sst_dev_dataloader,
                                                para_dev_dataloader,
                                                sts_dev_dataloader,
                                                model, device)

        #====add
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc_para/train",paraphrase_accuracy, epoch)
        writer.add_scalar("corr_sts/train",sts_accuracy, epoch)
        writer.add_scalar("acc_sentiment/train",sentiment_accuracy, epoch)
        writer.add_scalar("acc_para/dev",paraphrase_accuracy_dev, epoch)
        writer.add_scalar("corr_sts/dev",sts_accuracy_dec, epoch)
        writer.add_scalar("acc_sentiment/dev",sentiment_accuracy_dev, epoch)

        avg_dev_score=paraphrase_accuracy_dev+sentiment_accuracy_dev+sts_accuracy_dec

        if avg_dev_score > best_dev_acc:
            best_dev_acc = avg_dev_score
            save_model(model, optimizer, args, config, args.filepath)

        print(
            f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc para :: {paraphrase_accuracy :.3f},train acc sts :: {sts_corr :.3f},train acc sentiment :: {sentiment_accuracy :.3f},dev acc para:: {paraphrase_accuracy_dev :.3f},dev acc sts:: {sts_corr_dev :.3f},dev acc sentiment:: {sentiment_accuracy_dev :.3f}")


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")
    parser.add_argument("--run_path", type=str, default="bert_run") 

    # hyper parameters
    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'  # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)
