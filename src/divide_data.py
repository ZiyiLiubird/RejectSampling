import json
import os


data_path1 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', '7days30k_2.json')

raw_data1 = json.load(open(data_path1, "r"))
print(len(raw_data1))
batch = 467

save_path = os.path.join('/alg_vepfs/public/datasets/joyland/7days/', '3736_2sample')
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(8):
    data = raw_data1[i*batch: (i+1)*batch]
    save_path1 = os.path.join(save_path, f'7days3k_{i}.json')
    with open(save_path1, mode='w') as f:
        json.dump(data, f, indent=4)

