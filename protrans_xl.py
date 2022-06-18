# use xl encode protein sequences
from transformers import TFT5EncoderModel, T5Tokenizer
import numpy as np
import pandas as pd
import re
import gc
import os
from Bio import SeqIO

tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )

model = TFT5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", from_pt=True)

gc.collect()

dire = u'/home/khld071/python-virtual-environments/sub_seqs'
id_df = pd.read_csv('/projects/qbio/bifo/ProtProduction/khld071/protransxl_output/xl_clean_output.csv',low_memory=False)
unique_id_list = id_df['id'].tolist()
i = 0
names = locals()
length = 1024
trail_ids = []
feature_df = pd.DataFrame(columns = ['id'])
for index in range(length):
    names['xl_' + str(index)] = []
for files in os.listdir(dire):
 
    #trail_ids = []
    #feature_df = pd.DataFrame(columns = ['id'])
    #for index in range(length):
        #names['xl_' + str(index)] = []

    for record in SeqIO.parse(dire + '/' + files,'fasta'):
        if str(record.id) in unique_id_list:
            s = re.sub(r'[UZOB]','X',str(record.seq))
            s = re.sub(r'[^GAVLIPFYWSTCMNQDEKRHX]','',s)
            sequence = [' '.join(list(s))]
            ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True, return_tensors="tf")
            input_ids = ids['input_ids']
            attention_mask = ids['attention_mask']
            try:
                embedding = model(input_ids)
            except:
                continue
        
            trail_ids.append(str(record.id))
            embedding = np.asarray(embedding.last_hidden_state)
            attention_mask = np.asarray(attention_mask)

            seq_len = (attention_mask[0] == 1).sum()
            seq_emd = embedding[0][:seq_len-1]
            seq_emd_mean = np.mean(seq_emd,0).tolist()    
            for ind in range(length):
                try:
                    names['xl_' + str(ind)].append(seq_emd_mean[ind])
                except:
                    names['xl_' + str(ind)].append(None)
            
for y in range(length):
    feature_df['xl_' + str(y)] = names['xl_' + str(y)]
feature_df['id'] = trail_ids
output_path = '/projects/qbio/bifo/ProtProduction/khld071/protransxl_output/protrans_xl.csv'
feature_df.to_csv(output_path, sep = ',', index = False, header = True)


    
    
