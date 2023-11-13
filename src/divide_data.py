import json
import os

data_path = '/alg_vepfs/public/datasets/joyland/7days/7days.json'
raw_data = json.load(open(data_path, "r"))

save_path1 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', '7days15k_1.json')
save_path2 = os.path.join( '/alg_vepfs/public/datasets/joyland/7days/', '7days15k_2.json')

data = raw_data[:30000]
with open(save_path1, mode='w') as f:
    json.dump(data[:15000], f, indent=4)
with open(save_path2, mode='w') as f:
    json.dump(data[15000:], f, indent=4)

