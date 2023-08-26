mport json
import os
import torch
import re
import time
from transformers import AutoTokenizer, AutoModel, AutoConfig
def encode_instruct(web, instruct):
    desc = "用户的需求是[%s]。请问在网页上执行什么操作" % (instruct)
    return desc


def encode_response(web, instruct, key_values):
    desc = "用户使用的网站是[web]。在网页上" % (web)
    for key, value in key_values.items():
        action = ""
        print(key, value)
        if value["action"] == "点击":
            action = "[点击]名称为[%s]的[%s]" % (key, value["dom_type"])
        elif value["action"] == "输入":
            action = "[输入]值[%s]进名称为[%s]的[%s]" % (value["value"], key, value["dom_type"])
        desc += action + ";"
    return desc


def decode_response(desc):
    response = {}
    # print("desc: " + desc)
    actions = desc.split("。")[1]
    # print("actions: "+ actions)
    regex = r"(\[.*?\])"
    for action in actions.split(";"):
        matches = re.findall(regex, action)
        if len(matches) == 4:
            response[matches[2][1:-1]] = {'dom_type':matches[3][1:-1],
                                          'value':matches[1][1:-1],
                                          'action':'输入'}
        elif len(matches) == 3:
            response[matches[1][1:-1]] = {'dom_type':matches[2][1:-1],
                                          'value':'',
                                          'action':'点击'}
    return response


local_model_path = "./pre_trained_model/chatglm"

# 加载 Checkpoint
config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
config.pre_seq_len = 128

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(local_model_path, trust_remote_code=True, config=config).half()
model = model.to("cuda:0")

# 本次微调得到的glm权重
CHECKPOINT_PATH = "./ptuning/output/2023S-T1-A-Data-chatglm-6b-ft-1e-4/checkpoint-5000"


fine_tune_model_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")
prefix_state_dict = torch.load(fine_tune_model_path)

new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
#print(new_prefix_state_dict)
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict, strict=False)

# 测试是否部署完成
response, history = model.chat(tokenizer, '''用户的需求是[查询广东生物医药行业科创板的首创环保的信息]。请问在网页上执行什么操作？''', history=[])
print(response)
print(decode_response(response))

from tqdm import tqdm
LABEL_PATH = "./data/2023S-T1-A-Data/"
print("label_path:", LABEL_PATH)
instruction_testA = json.load(open(LABEL_PATH + 'instruction_testB.json', encoding='utf-8'))
submit_prediction = list()


for idx in tqdm(range(len(instruction_testA))):
    page_predictions = []
    page_source = instruction_testA[idx]['page_source']
    for index, instructions in enumerate(instruction_testA[idx]['instruction']):
        instruct = {}
        query = encode_instruct(page_source, instructions)
        response, history = model.chat(tokenizer, query)
        key_value = ""
        try:
            key_value = decode_response(response)
        except Exception as e:
            print("instruction:[%s] query:[%s] response:[%s]"% \
                  (instructions, query, response))
            print(e)
        print(key_value)
        instruct['instruction'] = instructions
        instruct['key-value'] = key_value
        page_predictions.append(instruct)
        #c = input("??")

    submit_prediction.append({
        'page_source': page_source,
        'instruction_detail': page_predictions
    })

print("submit_prediction [0]:", submit_prediction[0])
#submit_prediction = set(submit_prediction)
with open('submission.json', 'w') as up:
    json.dump(submit_prediction, up,ensure_ascii=False)
