import copy
import torch
from torch import nn
import timm
from models.basic_lora import LoRAViT

def get_vit(args, pretrained=False):
    name = args['vit_type'].lower()
    rank = args['lora_rank']

    if name == 'vit-b-p16':
        vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
    elif name == 'vit-b-p16-mocov3':
        vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        model_dict = torch.load('mocov3-vit-base-300ep.pth', weights_only=False)
        vit.load_state_dict(model_dict['model'], strict=True)
    elif name == 'vit-b-p16-dino':
        vit = timm.create_model('vit_base_patch16_224.dino', pretrained=pretrained, num_classes=0)
    elif name == 'vit-b-p16-mae':
        vit = timm.create_model('vit_base_patch16_224.mae', pretrained=pretrained, num_classes=0)
    else:
        raise ValueError(f'Model {name} not supported')

    # Remove original head (already num_classes=0, but ensure)
    if hasattr(vit, 'head') and isinstance(vit.head, nn.Linear):
        vit.head = nn.Identity()

    lora_type = args['lora_type']
    if lora_type == "full":
        model = vit

    elif lora_type == "basic_lora":
        model = LoRAViT(vit, r=rank, use_dora=False)
        model.activate_lora_params(include_norms=False)

    else:
        raise ValueError(f"LoRA type {lora_type} not supported")
    
    return model


class ContinualLinear(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes, bias=False)])
        self.head_weights = nn.Parameter(torch.ones(nb_classes))
        self.current_output_size = nb_classes

    def update(self, nb_classes, freeze_old=True):
        # Create new head
        new_head = nn.Linear(self.embed_dim, nb_classes, bias=False)
        
        # Freeze old heads if requested
        if freeze_old:
            for head in self.heads:
                for param in head.parameters():
                    param.requires_grad = False
        
        # Add new head
        self.heads.append(new_head)
        
        # Update head weights
        new_head_weights = nn.Parameter(torch.ones(self.current_output_size + nb_classes))
        with torch.no_grad():
            new_head_weights[:self.current_output_size] = self.head_weights
            new_head_weights[self.current_output_size:] = 1.0
        
        self.head_weights = new_head_weights
        self.current_output_size += nb_classes

    def forward(self, x):
        # Process all heads in parallel using list comprehension
        outputs = [head(x) for head in self.heads]
        # Concatenate outputs along the class dimension
        combined = torch.cat(outputs, dim=1)
        # Apply learned weights
        return combined * self.head_weights


class BaseNet(nn.Module):
    def __init__(self, cfg, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_vit(cfg, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.feature_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class FinetuneIncrementalNet(BaseNet):

    def __init__(self, cfg, pretrained):
        super().__init__(cfg, pretrained)

    @property
    def feature_dim(self):
        return self.convnet.feature_dim

    def update_fc(self, nb_classes, freeze_old=True):
        if self.fc is None:
            self.fc = ContinualLinear(self.feature_dim, nb_classes)
        else:
            self.fc.update(nb_classes, freeze_old)

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        return out