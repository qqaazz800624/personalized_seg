#%%

#from manafaln.models import mednext_base
from custom.mednext import mednext_base
import torch

model_mednext = mednext_base(spatial_dims=3,
                     in_channels=1,
                     out_channels=16,
                     kernel_size=3,
                     filters=32,
                     deep_supervision=True,
                     use_grad_checkpoint=True,
                     )

model_mednext.eval()

ckpt_path = "/home/u/qqaazz800624/personalized_seg/mri_seg/MRI/bgz2zat3/checkpoints/best_model.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

if 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt

new_state_dict = {}
for k, v in state_dict.items():
    if "loss_fn" in k:
        continue
    new_key = k.replace('model.', '')
    new_state_dict[new_key] = v

model_mednext.load_state_dict(new_state_dict)

ckpt['state_dict'] = new_state_dict

new_ckpt_path = "/home/u/qqaazz800624/personalized_seg/mri_seg/MRI/bgz2zat3/checkpoints/new_best_model.ckpt"
torch.save(ckpt, new_ckpt_path)


#%%









#%%






#%%