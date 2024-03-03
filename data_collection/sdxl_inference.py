import pandas as pd
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DiffusionPipeline, StableDiffusionXLPipeline, AutoPipelineForText2Image
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS
import torch
import os
import matplotlib
import tqdm
from accelerate import PartialState
import datetime


def extract_file_ids(file_list: list) -> list:
    file_ids = list()
    for i, img_name in enumerate(file_list):
        try:
            file_ids.append(int(img_name.split('.')[0]))
        except:
            print('bad id:', img_name)
    return file_ids


def create_batch_generator(data,
                           column,
                           batch_size,
                           id_column='ranking_id',
                           generated_files_ids=None,
                           **kwargs):
    prompt_add = ''
    if generated_files_ids:
        data = data[~data[id_column].isin(generated_files_ids)]
    total_rows = len(data)
    num_batches = total_rows // batch_size
    prompt_list = data[column].tolist()
    id_list = data[id_column].tolist()
    print(len(prompt_list), ' imgs to generate, num batches: ', num_batches)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch = prompt_list[start_idx:end_idx]
        batch_ids = id_list[start_idx:end_idx]
        
        batch = [p + prompt_add for p in batch]
        pipe_args = {**kwargs, **{"prompt": batch}}
        yield batch_ids, pipe_args

    # Yield the last batch (which might have fewer elements)
    if total_rows % batch_size != 0:
        # generator = [torch.Generator("cuda").manual_seed(i) for i in range(batch_size)]
        start_idx = num_batches * batch_size
        batch = prompt_list[start_idx:]
        
        batch = [p + prompt_add for p in batch]
        batch_ids = id_list[start_idx:]
        pipe_args = {**kwargs, **{"prompt": batch}}
        yield batch_ids, pipe_args

### sdxl_t
img_dir = '/home/aokulikov/proj_dpo_shared/data/generated_imgs/sdxl_t_4_coyo_llava_prep'

df_path = '/home/aokulikov/proj_dpo_shared/data/datasets'

# df prep
df = pd.read_csv('/home/aokulikov/proj_dpo_shared/data/imgs/coyo_prepared/llava_captions_prepared.csv')


#generated files
file_list = os.listdir(img_dir)
file_ids = extract_file_ids(file_list)

#sdxl_t batch_generator
batch_generator = create_batch_generator(data=df,
                                         column='llava_caption',
                                         id_column='img_id',
                                         batch_size=8,
                                         num_inference_steps=4,
                                         generated_files_ids=file_ids,
                                         guidance_scale=0,
                                         high_noise_frac=None,
                                         height=768,
                                         width=768)

batch_list = list()
for b in tqdm.tqdm(batch_generator):
    batch_list.append(b)

# sdxl_t
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo",
                                                 torch_dtype=torch.float16,
                                                 variant="fp16")


distributed_state = PartialState()
pipe.to(distributed_state.device)
print('num batches', len(batch_list))
s = datetime.datetime.now()
with distributed_state.split_between_processes(batch_list) as batch_sublist:
    for batch_ids, batch_args in batch_sublist:
        print(batch_ids)
        print(batch_args)
        img_list = pipe(**batch_args).images
        for img, id in zip(img_list, batch_ids):
            img.save(os.path.join(img_dir, f'{id}.jpeg'))
        
        
print(f'{len(batch_list)} batches took {datetime.datetime.now() - s}')