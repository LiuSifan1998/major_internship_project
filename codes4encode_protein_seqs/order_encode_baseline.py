# order encode
from Bio import SeqIO
import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import LabelEncoder

seq,ids = [],[]
max_len = 1280
names = locals()
dire = u'/home/khld071/python-virtual-environments/2_subset'
aa_list = ['G','A','V','L','I','P','F','Y','W','S','T','C','M','N','Q','D','E','K','R','H','X',' ']
for x in range(max_len):
    names['v'+ str(x)] = []

for files in os.listdir(dire):
    for record in SeqIO.parse(dire + '/' + files,'fasta'):
        s = re.sub(r'[UZOB]','X',str(record.seq))
        s = re.sub(r'[^GAVLIPFYWSTCMNQDEKRHX]','',s)
        if len(s) > max_len:
            continue
        else:
            seq.append(s)
            ids.append(str(record.id))
            
def string_to_array(string):
    string = re.sub(r'[UZOB]','X',string)
    my_array = np.array(list(string))
    return my_array

label_encoder = LabelEncoder()
label_encoder.fit(np.array(aa_list))


def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    for i in range(22):
        if i == 20:
            float_encoded[float_encoded == i] = 0.00 # X
        if i == 21:
            float_encoded[float_encoded == i] = 3.00 # ' '
        else:
            float_encoded[float_encoded == i] = (i + 1) * 0.10
            
    return float_encoded

for sequences in seq:
    if len(sequences) < max_len:
        test_sequence = sequences + ' '*(max_len - len(sequences))
        
    else:
        test_sequence = sequences

    l = ordinal_encoder(string_to_array(test_sequence)).tolist()
    
    for ind in range(max_len):
        names['v'+ str(ind)].append(l[ind])

df = pd.DataFrame(columns=['id'])

for index in range(max_len):
    df['v'+str(index)] = names['v'+str(index)]
df['id'] = ids

other_df = pd.read_csv('/projects/qbio/bifo/ProtProduction/khld071/order_encode.csv',low_memory=False)
mergedf = other_df.append(df)

output_path = '/projects/qbio/bifo/ProtProduction/khld071/order_encode_output/order_encode.csv'
mergedf.to_csv(output_path, sep = ',', index = False, header = True)

