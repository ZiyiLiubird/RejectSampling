import json
import os
import copy

sft_data_path = '/alg_vepfs/public/LZY/sft_data/2023-11-15'
samplel_data_path = '/alg_vepfs/public/LZY/sample_data/2023-11-15'

new_sft_data_path = os.path.join(sft_data_path, 'processed_sft_data.json')

processed_sft_data_list = []

def solve():

    for i in range(8):
        sft_data = json.load(open(os.path.join(sft_data_path, f"sft{i}.json"), "r"))
        sample_data = json.load(open(os.path.join(samplel_data_path, f"sample_reward{i}.json"), "r"))
        cnt = 0
        for sample_conv in sample_data:
            sample_conv_piece = sample_conv['conversations']
            for index, block in copy.copy(enumerate(sample_conv_piece)):
                if block['from'] == 'sample':
                    sample_conv_piece.pop(index)

        for sft_conv, sample_conv in zip(sft_data, sample_data):
            if len(sft_conv['conversations']) != len(sample_conv['conversations']):
                continue
            character_name = sft_conv['character_name']
            user_name = sft_conv['user_name']
            sft_conv_piece, sample_conv_piece = sft_conv['conversations'], sample_conv['conversations']
            output_data_piece = {"character_name": character_name, "user_name": user_name, 'conversations': []}
            for index, block in enumerate(sft_conv_piece):
                if block['from'] == 'instruction':
                    block['train'] = False
                    output_data_piece['conversations'].append(block)
                elif (block['from'] == character_name and index == 1) or block['from'] == 'Me':
                    block['train'] = False
                    output_data_piece['conversations'].append(block)
                elif block['from'] == character_name:
                    temp_data_piece = copy.deepcopy(output_data_piece)
                    temp_data_piece['id'] = cnt
                    cnt += 1
                    block['train'] = True
                    temp_data_piece['conversations'].append(block)
                    processed_sft_data_list.append(temp_data_piece)
                    if sample_conv_piece[index]['from'] != character_name:
                        break
                    assert sample_conv_piece[index]['from'] == character_name, print(index, len(sft_conv_piece))
                    sample_conv_piece[index]['train'] = False
                    output_data_piece['conversations'].append(sample_conv_piece[index])


    with open(new_sft_data_path, 'w') as file:
        json.dump(processed_sft_data_list, file, indent=4)



if __name__ == '__main__':
    data = json.load(open(os.path.join(new_sft_data_path), "r"))
    solve()