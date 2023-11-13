import json
import os


data_path1 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', '7days15k_1.json')
data_path2 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', '7days15k_2.json')

raw_data1 = json.load(open(data_path1, "r"))
raw_data2 = json.load(open(data_path2, "r"))

batch = 3750

for i in range(4):
    data = raw_data1[i*batch: (i+1)*batch]
    save_path1 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', f'7days30k_{i}.json')
    with open(save_path1, mode='w') as f:
        json.dump(data, f, indent=4)

for i in range(4):
    data = raw_data2[i*batch: (i+1)*batch]
    save_path1 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', f'7days30k_{i+4}.json')
    with open(save_path1, mode='w') as f:
        json.dump(data, f, indent=4)


