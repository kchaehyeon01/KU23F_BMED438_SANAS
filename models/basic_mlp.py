import torch
import torch.nn as nn
import torch.nn.functional as F

class basic_MLP(nn.Module):
    """
        - input : BxNx3 (batchsize/#ofpointspersample/xyz)
    """
    def __init__(self):
        super(basic_MLP, self).__init__()
        self.b_size = 128
        self.n_size = 2048 # 데이터에서 알아서 파악하는 걸로 수정 필요, 원래 논문에서는 512 사용함!

        self.linear1 = nn.Linear(3, 64)     # BxNx3 -> BxNx64
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(64, 128)   # BxNx64 -> BxNx128
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, 1024) # BxNx128 -> BxNx1024

         # fully connected : (512,256,9) => (3x3)
        self.fc1 = nn.Linear(1024, 512)
        self.reluf1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 9)  # => (3x3)
        self.reluf2 = nn.ReLU()
        
    
    def forward(self, x): 

        x = x.view(-1, 3)    # 128x2048x3 -> 262144x3
        x = self.linear1(x)  # 262144x3 -> 262144x64
        x = self.relu1(x)
        x = self.linear2(x)  # 262144x64 -> 262144x128
        x = self.relu2(x)    
        x = self.linear3(x)  # 262144x128 -> 262144x1024
        x = torch.max(x, dim=0)  # maxpooling : 262144x1024 -> 1024
        x = x[0] # torch.max는 index까지 돌려주므로, 데이터만 받아야 함
        x = self.fc1(x)       # 1024 -> 512
        x = self.reluf1(x)
        x = self.fc2(x)      # 512 -> 9
        
        x = x.view(3,3)

        return x
        