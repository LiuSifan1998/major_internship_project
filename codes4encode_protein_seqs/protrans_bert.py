import tensorflow as tf
from transformers import TFBertModel, BertTokenizer,BertConfig
import re
import numpy as np
import pandas as pd
import os
from Bio import SeqIO

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )

model = TFBertModel.from_pretrained("Rostlab/prot_bert", from_pt=True)

dire = u'/home/khld071/python-virtual-environments/sub_seqs'


names = locals()
length = 1024

for files in os.listdir(dire):

    trail_ids = []
    feature_df = pd.DataFrame(columns = ['id'])
    for index in range(length):
        feature_df['bert_' + str(index)] = []
        names['bert_' + str(index)] = []

    for record in SeqIO.parse(dire + '/' + files,'fasta'):
        trail_ids.append(str(record.id))
        sequence = ' '.join(list(str(record.seq)))
        sequence = re.sub(r"[UZOB]", "X", sequence)
        ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True, return_tensors="tf")
        input_ids = ids['input_ids']
        attention_mask = ids['attention_mask']
        try:
            embedding = model(input_ids)[0]
        except:
            continue

        embedding = np.asarray(embedding)
        attention_mask = np.asarray(attention_mask)
        seq_len = (attention_mask[0] == 1).sum()
        seq_emd = embedding[0][1:seq_len-1]
        seq_emd_mean = np.mean(seq_emd,0)
        for ind in range(length):
            try:
                names['bert_' + str(ind)].append(seq_emd_mean[ind])
            except:
                names['bert_' + str(ind)].append(None)
                

    for y in range(length):
        feature_df['bert_' + str(y)] = names['bert_' + str(y)]

    feature_df['id'] = trail_ids
    
    output_path = '/projects/qbio/bifo/ProtProduction/khld071/protransbert_output/protrans_bert' + files + '.csv'
    feature_df.to_csv(output_path, sep = ',', index = False, header = True)
