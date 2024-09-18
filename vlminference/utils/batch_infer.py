import argparse
import json
import os
import sys
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(main_dir)

from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm

# 批处理
def collate_fn(batch):
    id_list = [item['id'] for item in batch]
    query_list = [item['query'] for item in batch]
    imgs_list = [item['imgs'] for item in batch]
    return id_list, query_list, imgs_list

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default="InternVL2", help='The name of the model.')
parser.add_argument('--model_path', type=str, default="OpenGVLab/InternVL2-8B", help='The local directory or the repository id of the model.')
parser.add_argument('--batch_size', type=int, default=8, help='The batch size.')
parser.add_argument('--input_file', type=str, default=None, help='The path of the input file.')
parser.add_argument('--output_file', type=str, default=None, help='The path of the output file.')
config = EasyDict(vars(parser.parse_args()))
print(config)

# 加载数据
print("***************Data Loading Start****************")
assert config.input_file is not None, "Please specify the input file."
assert config.output_file is not None, "Please specify the output file."
dataset = []
with open(config.input_file, 'r') as f:
    for line in f:
        cur_json = json.loads(line.strip())
        if "imgs" not in cur_json.keys():
            cur_json["imgs"] = None
        elif cur_json["imgs"] is None:
            pass
        elif len(cur_json["imgs"]) == 0:
            cur_json["imgs"] = None
        dataset.append(cur_json)
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
print(f"[Dataset]>>> There are total {len(dataset)} samples.")
print("***************Data Loading Done*****************","\n")

# 加载模型
print("***************Model Loading Start***************")
if config.model_name == "InternVL2":
    os.environ['MODEL_ENV'] = 'internvl2'
    from vlminference.models import InternVL2ForInfer as ModelForInfer
else:
    raise NotImplementedError(f"The model {config.model_name} is not implemented.")
infer_engine = ModelForInfer(config.model_path)
print(f"[Model]>>> {config.model_name} for inference.")
print("***************Model Loading Done****************","\n")

print("***************Inference Start*******************")
output_list = []
for batch_data in tqdm(dataloader):
    id_list, query_list, imgs_list = batch_data
    response_list = infer_engine.batch_infer(query_list, imgs_list)
    for cur_id, cur_response in zip(id_list, response_list):
        cur_output = {
            "id": cur_id,
            "response": cur_response
        }
        output_list.append(cur_output)
    
    with open(config.output_file, 'w') as f:
        for cur_output in output_list:
            f.write(json.dumps(cur_output, ensure_ascii=False) + "\n")
# print(output_list)
print("***************Inference Done********************","\n")
