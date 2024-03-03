from torchvision import transforms
import torchvision

from datasets import load_dataset, Dataset, Image

import pandas as pd
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16).to("cuda")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

url = '/home/aokulikov/proj_dpo_shared/data/imgs/coyo_prepared/train'
result_path = '/home/aokulikov/proj_dpo_shared/data/imgs/coyo_prepared'
metadata = pd.read_csv(f'{url}/metadata.csv')


img_list = (url + '/' + metadata['file_name']).tolist()
file_name_list = metadata['file_name'].tolist()

ds = Dataset.from_dict({"image": img_list, 'file_name': file_name_list}).cast_column("image", Image())
batch_size = 32

def predict_caption(batch):
    prompt = "<image>\nUSER: Create a caption that best describes the image. Make it detailed, short and effective for a diffusion model.\nASSISTANT:"
    img_list = batch['image']
    prompt_list = [prompt] * len(batch['image'])
    try:
        inputs = processor(text=prompt_list, images=img_list, return_tensors="pt").to("cuda", torch.float16)
        generate_ids = model.generate(**inputs, max_length=80)
        result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result_dict = {'captions': result}
    except:
        result_dict = {'captions': ['no caption'] * len(batch['image'])}
    return result_dict


captions_dict = ds.map(
    predict_caption,
    batched=True,
    batch_size=batch_size
)
captions_df = pd.DataFrame(data={'file_name': captions_dict['file_name'], 'llava_caption': captions_dict['captions']})
captions_df['img_id'] = captions_df['file_name'].apply(lambda x: x.split('.')[0])
captions_df.to_csv(f'{result_path}/llava_captions_v2.csv', index=False)