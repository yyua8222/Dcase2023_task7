import sys

sys.path.append("src")

import os
import numpy as np

import argparse
import yaml
import torch
from latent_diffusion.models.ddpm import LatentDiffusion
from tqdm import tqdm
import torchaudio
import json


config_root = "configs"

big_config = os.path.join(config_root, "big.yaml")
big_config = yaml.load(
    open(big_config, "r"), Loader=yaml.FullLoader
)

small_config = os.path.join(config_root, "small.yaml")
small_config = yaml.load(
    open(small_config, "r"), Loader=yaml.FullLoader
)


def get_big():
    latent_diffusion = LatentDiffusion(**big_config["model"]["params"])
    PATH = "model_logs/clap_submit.ckpt"
    state_dict = torch.load(PATH)["state_dict"]
    try:
        transformer = state_dict["transform.kernel"]
        del state_dict["transform.kernel"]
    except:
        pass
    latent_diffusion.load_state_dict(state_dict)
    return latent_diffusion
def get_small():
    latent_diffusion = LatentDiffusion(**small_config["model"]["params"])
    PATH = "model_logs/dcase_submit.ckpt"
    state_dict = torch.load(PATH)["state_dict"]
    try:
        transformer = state_dict["transform.kernel"]
        del state_dict["transform.kernel"]
    except:
        pass
    latent_diffusion.load_state_dict(state_dict)
    return latent_diffusion
# def process():


big_model = get_big().cuda()
big_model = torch.compile(big_model)
big_model.eval()
small_model = get_small().cuda()
small_model = torch.compile(small_model)
small_model.eval()
# name = big_model.get_validation_folder_name()


# label_list = ["foot","cough","dog","keyboard","gun"]
# caption_list = ["foot steps","a man cough","a dog bark","someone using keyboard","gun shot"]
# limit_list = [0.05,0.2,0.37,0.2,0.2]
# gen_list = [3,3,3,3,3]
# scale_list = [2.5,2.5,2.5,1.0,1.0]

label_list = ['dog_bark', 'footstep', 'gunshot', 'keyboard', 'moving_motor_vehicle', 'rain', 'sneeze_cough']
caption_list = ["a dog bark","foot steps","gun shot","someone using keyboard","a moving motor","rain","a man cough"]
limit_list = [0.37,0.05,0.2,0.2,0.75,0.2,0.2]
gen_list = [3,3,3,3,10,3,3]
scale_list = [2.5,2.5,1.0,1.0,3,1.0,2.5]
target_list = [-1,-1,-1,-1,0,-1,-1]
model_list = [0,0,0,0,1,1,0]



# label_list = ["motor"]
# caption_list = ["a moving motor"]


def generate_sound(id,quantity = 100,model = big_model):

    clap = model.cond_stage_model
    attampt = 2
    num = gen_list[id]
    label = label_list[id]
    g_caption = caption_list[id]
    limit = limit_list[id]
    sam = 0
    target = target_list[id]
    scale = scale_list[id]

    result_list = []



    for j in tqdm(range(quantity)):
        with torch.no_grad():
#             model.logger_save_dir = f"/mnt/fast/nobackup/scratch4weeks/yy01071/audioLDM/audioLDM_decase/results/temp{attampt}/" + label
#             model.logger_project = ""
#             model.logger_version = ""
            model.cond_stage_key = "text"
            model.cond_stage_model.embed_mode = "text"
            count = 0
            batch = [torch.rand(1,400,64), torch.rand(1,400,512), torch.rand(1,527), (f"{str(sam)}.wav",),torch.rand(1,64000),(g_caption,)]

            
            saved,waveform,result = model.generate_sample(
                [batch],
                name=label,
                unconditional_guidance_scale=scale,
                ddim_steps=model.evaluation_params["ddim_sampling_steps"],
                n_gen=num,
                limit=limit,
                target=target
            )
            count += 1
            change_limit = 0
            while not saved:
                if count>=3:
                    # limit = limit -0.05
                    change_limit +=1
                    count=0
                    # limit = limit-0.01
                saved,waveform,result = model.generate_sample(
                [batch],
                name=label,
                unconditional_guidance_scale=scale,
                ddim_steps=model.evaluation_params["ddim_sampling_steps"],
                n_gen=num,
                limit=limit,
                target=target,
                change_limit = change_limit
                )
                count+=1

            if target>=0:
                sam+=1
                target +=1
            else: 
                low = False
                waveforms = torch.cat([waveform, waveform])
                txt = [g_caption, g_caption]
                similarity = clap.cos_similarity(waveforms.cuda(), txt)
                while similarity[0] != similarity[1]:
                    print("output not equal!")
                    similarity = clap.cos_similarity(waveforms.cuda(), txt)

                score = similarity[0].cpu().detach().numpy() + 0
                if score < limit:
                    low = True
                while low:
                    low = False
                    # print("output too low")
                    saved, waveform,result = model.generate_sample(
                        [batch],
                        name=label,
                        unconditional_guidance_scale=scale,
                        ddim_steps=model.evaluation_params["ddim_sampling_steps"],

                        n_gen=num,
                        limit=limit,
                    )
                    while not saved:
                        saved, waveform,result = model.generate_sample(
                            [batch],
                            name=label,
                            unconditional_guidance_scale=scale,
                            ddim_steps=model.evaluation_params["ddim_sampling_steps"],
                            n_gen=num,
                            limit=limit,
                        )

                    waveforms = torch.cat([waveform, waveform])
                    txt = [g_caption, g_caption]
                    similarity = clap.cos_similarity(waveforms.cuda(), txt)
                    while similarity[0] != similarity[1]:
                        # print("output not equal!")
                        similarity = clap.cos_similarity(waveforms.cuda(), txt)

                    score = similarity[0].cpu().detach().numpy() + 0
                    if score < limit:
                        low = True
                sam+=1

            result_list.append(result)
            
def samplying_sound(id,quantity = 100):

    model_id = model_list[id]

    if model_id==0:
        generate_sound(id,quantity=quantity,model = small_model)

    if model_id==1:
        generate_sound(id,quantity=quantity,model = big_model)

for i in range(len(label_list)):
    samplying_sound(i,quantity=5)
