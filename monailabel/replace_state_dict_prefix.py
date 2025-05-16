#%%

import torch
from monai.networks.nets import SegResNet

segresnet = SegResNet(blocks_down = [1, 2, 2, 4],
                      blocks_up = [1, 1, 1],
                      init_filters = 16,
                      in_channels = 1,
                      out_channels = 3,
                      dropout_prob = 0.0)
segresnet.eval()

#%%
import torch

ckpt_path = "/home/u/qqaazz800624/medaseg/pinkcc/PINKCC/49cfo62r/checkpoints/best_model.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

print(ckpt["state_dict"])
#%%

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

new_state_dict = {}
for k, v in state_dict.items():
    if "loss_fn" in k:
        continue
    new_key = k.replace('model.backbone._orig_mod.', '')
    new_state_dict[new_key] = v

segresnet.load_state_dict(new_state_dict)

ckpt['state_dict'] = new_state_dict

#%%
new_ckpt_path = "/home/u/qqaazz800624/personalized_seg/monailabel/model/pinkcc_segresnet.pt"
torch.save(ckpt, new_ckpt_path)


#%%

import torch

ckpt_path = "/home/u/qqaazz800624/personalized_seg/monailabel/model/pinkcc_segresnet.pt"

ckpt = torch.load(ckpt_path, map_location='cpu')
ckpt



#%%


ckpt['state_dict'].keys()



#%%



