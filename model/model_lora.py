import torch
from torch import nn


class LoRA(nn.Module):
    """
    origin: h = Wx 
    LoRA: h = W'x = (W + BA)x = Wx + BAx
    """
    def __init__(self, in_features, out_features, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank    # rank of lora
        self.A = nn.Linear(in_features, rank, bias=False)  
        self.B = nn.Linear(rank, out_features, bias=False)
    
        # init
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()
        
    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model: nn.Module, rank: int=8):
    """
        Apply LoRA on Linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
           lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
           setattr(module, "lora", lora)
           original_forward = module.forward    # forward of original Linear
           
           def forward_with_lora(x, layer1=original_forward, layer2=lora):
               return layer1(x) + layer2(x)
           
           module.forward = forward_with_lora


def save_lora(model: nn.Module, path: str):
    """
        Save A and B matrices of LoRA layers
        
        e.g. 
        {
            linear1.lora.A.weight: Tensor
            linear1.lora.B.weight: Tensor
            linear2.lora.A.weight: Tensor
            linear2.lora.B.weight: Tensor
        }
    """
    state_dict = {}   
    
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f"{clean_name}.lora.{k}": v for k, v in module.lora.state_dict().items()} 
            state_dict.update(lora_state)

    torch.save(state_dict, path)


def load_lora(model: nn.Module, path: str):
    state_dict = torch.load(path, map_location=model.device)  # load lora weight
    state_dict = { (k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            # linear1.lora.A.weight -> A.weight
            lora_state = {k.replace(f"{name}.lora.", ""): v for k, v in state_dict.items() if f"{name}.lora." in k}
            module.lora.load_state_dict(lora_state)


if __name__ == "__main__":
    
    class TestModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.linear1 = nn.Linear(128, 128)
            self.linear2 = nn.Linear(128, 128)
            self.linear3 = nn.Linear(128, 10)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x
    
    # test LoRA
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rank = 8
    lora_save_path = "test_lora_weights.pth"
    
    model = TestModel().to(device)
    model.device = device
    
    # TestModel(
    #     (linear1): Linear(in_features=128, out_features=128, bias=True)
    #     (linear2): Linear(in_features=128, out_features=128, bias=True)
    #     (linear3): Linear(in_features=128, out_features=10, bias=True)
    # )
    print("========= Model Architecture ========")
    print(model)
    
    
    # TestModel(
    #   (linear1): Linear(
    #     in_features=128, out_features=128, bias=True
    #     (lora): LoRA(
    #       (A): Linear(in_features=128, out_features=8, bias=False)
    #       (B): Linear(in_features=8, out_features=128, bias=False)
    #     )
    #   )
    #   (linear2): Linear(
    #     in_features=128, out_features=128, bias=True
    #     (lora): LoRA(
    #       (A): Linear(in_features=128, out_features=8, bias=False)
    #       (B): Linear(in_features=8, out_features=128, bias=False)
    #     )
    #   )
    #   (linear3): Linear(in_features=128, out_features=10, bias=True)
    # )
    print("========= Model Architecture (LoRA) ========")
    apply_lora(model, rank)
    print(model)

    # save and load
    save_lora(model, lora_save_path)
    load_lora(model, lora_save_path)
           

    