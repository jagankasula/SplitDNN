import torch
import torch.nn as nn
import torchvision.models as models
import warnings

warnings.filterwarnings('ignore')


device = "cpu:0"
candidate_layers = []

model = models.swin_s(pretrained=True)

#all_modules = list(model.features.children())

all_modules = list(model.features.children())

print(f'Total number of available layers: {len(all_modules)}')

for split_point in range(1, len(all_modules) + 1):

    try:
        left_modules = all_modules[:split_point]
        left_modules = nn.Sequential(*left_modules)
        left_model = left_modules.eval().to(device)

        right_modules = all_modules[split_point: -1]
        right_modules = nn.Sequential(*right_modules)
        right_model = right_modules.eval().to(device)


        # input = torch.randn(1, 3, 224, 224)

        # input =  torch.randn(1, 768, 14, 14)

        # PROFILING
        # with torch.profiler.profile(activities=[
        #     torch.profiler.ProfilerActivity.CPU,], with_flops = True
        #     ) as p:
        #     left_output = left_model(input)

        # print(p.key_averages().table(row_limit=-1))

        left_output = left_model(input)

        output = right_model(left_output.to(device))

        # print(output.shape)

        print(f'Success at split point # {split_point}')

        candidate_layers.append(split_point)

    except ValueError as vrr:
        print(f'------- Failed at split point # {split_point}')
        print(vrr)        
    except Exception as e:
        print(f'------- EXCEPTION -- Failed at split point # {split_point}')
        print(e)

print(candidate_layers)