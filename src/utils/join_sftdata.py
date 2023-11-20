import json
import os


data_path1 = os.path.join('/alg_vepfs/public/reject_sampling_dataset')

save_path = '/alg_vepfs/public/reject_sampling_dataset/joined/stfdata.json'

data_list = []

data = json.load(open(os.path.join(data_path1, "processed_sft_data.json"), "r"))
data_list.extend(data)


data1 = json.load(open(os.path.join(data_path1, "processed_sft_data1.json"), "r"))
data_list.extend(data1)


with open(save_path, 'w') as file:
    json.dump(data_list, file, indent=4)


# if __name__ == '__main__':
#     data = json.load(open(save_path, "r"))
#     print(len(data))

