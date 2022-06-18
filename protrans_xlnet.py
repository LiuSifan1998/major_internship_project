import tensorflow as tf
from transformers import TFXLNetModel, XLNetTokenizer,XLNetConfig
import re
import numpy as np
import pandas as pd
import os
from Bio import SeqIO

tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", do_lower_case=False )
xlnet_men_len = 1024
model = TFXLNetModel.from_pretrained("Rostlab/prot_xlnet", mem_len=xlnet_men_len, from_pt=True)

dire = u'/home/khld071/python-virtual-environments/sub_seqs'
id_df = pd.read_csv('/projects/qbio/bifo/ProtProduction/khld071/protransxlnet_output/xlnet_clean_output.csv',low_memory=False)
unique_id_list = id_df['id'].tolist()
i = 0
length = 1024
names = locals()
for index in range(length):
    names['xlnet_' + str(index)] = []
feature_df = pd.DataFrame(columns = ['id'])
trail_ids = []

for files in os.listdir(dire):

    #trail_ids = []
    #feature_df = pd.DataFrame(columns = ['id'])
    #for index in range(length):
    #    names['xlnet_' + str(index)] = []

    for record in SeqIO.parse(dire + '/' + files,'fasta'):
        if str(record.id) in unique_id_list: 
            s = re.sub(r'[UZOB]','X',str(record.seq))
            s = re.sub(r'[^GAVLIPFYWSTCMNQDEKRHX]','',s)
            sequence = [' '.join(list(s))]
            ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True, return_tensors="tf")
            input_ids = ids['input_ids']
            attention_mask = ids['attention_mask']
            try:
                output = model(input_ids,attention_mask=attention_mask,mems=None)
            except:
                continue
            trail_ids.append(str(record.id))
            embedding = output.last_hidden_state
            memory = output.mems
            embedding = np.asarray(embedding)
            attention_mask = np.asarray(attention_mask)
            seq_len = (attention_mask[0] == 1).sum()
            padded_seq_len = len(attention_mask[0])
            seq_emd = embedding[0][padded_seq_len-seq_len:padded_seq_len-2]
            seq_emd_mean = np.mean(seq_emd,0).tolist()
            for ind in range(length):
                try:
                    names['xlnet_' + str(ind)].append(seq_emd_mean[ind])
                except:
                    names['xlnet_' + str(ind)].append(None)                

for y in range(length):
    feature_df['xlnet_' + str(y)] = names['xlnet_' + str(y)]

feature_df['id'] = trail_ids
    
output_path = '/projects/qbio/bifo/ProtProduction/khld071/protransxlnet_output/protrans_xlnet.csv'
feature_df.to_csv(output_path, sep = ',', index = False, header = True)
