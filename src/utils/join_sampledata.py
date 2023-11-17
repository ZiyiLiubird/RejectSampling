import json
import os


data_path1 = os.path.join('/alg_vepfs/public/LZY/sample_data/2023-11-16')


batch = 467

save_path = '/alg_vepfs/public/LZY/sample_data/2023-11-16/sample_all.json'

data_list = []

for i in range(8):
    if i == 1:
        continue
    data = json.load(open(os.path.join(data_path1, f"sample{i}.json"), "r"))
    assert type(data) == list
    data_list.extend(data)


with open(save_path, 'w') as file:
    json.dump(data_list, file, indent=4)


# if __name__ == '__main__':
#     data = json.load(open(save_path, "r"))
#     print(len(data))

