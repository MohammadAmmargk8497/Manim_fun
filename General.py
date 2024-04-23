import torch
from torch import nn
from torch import Tensor
from torch import Tuple

d_k = 16
d_f = 48
d_v = 16
d_q = 16 

class KVSplitter(nn.Module) :
     def __init__(self):
        super(self).__init__()
        self.d_k = 16
        self.d_f = 48
        self.d_v = 16
        self.d_q = 16 
        self.query_proj = nn.Linear(self.d_f, self.d_q, bias=False)
        self.key_proj = nn.Linear(self.d_f, self.d_q, bias=False)
        self.bias = nn.Parameter(torch.rand(self.d_k).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(self.d_k, 1)
        self.value_proj = nn.Linear(self.d_f, self.d_v, bias = False )

     def forward(self, x, q):
         K = self.key_proj(torch.t(x))
         
         V = self.value_proj(torch.t(x))
        
         Q = self.query_proj(torch.t(q))
        
         return K, Q, V 


     
class AdditiveAttention(nn.Module):
      def __init__(self):
          super(self).__init__()
          self.d_w = 32
          self.W_1 = nn.Linear(d_q, self.d_w, bias = False )
          self.W_2 = nn.Linear(d_k, self.d_w, bias = False)  
          self.bias = nn.Parameter(torch.rand().uniform_(-0.1, 0.1))
          self.score_proj = nn.Linear(self.d_k, 1)
      
      def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
 
      
          score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
          attn = F.softmax(score, dim=-1)
          context = torch.bmm(attn.unsqueeze(1), value)
          return context, attn
      
        
     

if __name__ == '__main__':
    d_f = 48
    n_f = 128
    d_q = 48
    x = torch.randn(d_f, n_f)
    q = torch.randn(d_q, n_f)
    Att = KVSplitter()  
    out = Att(x,q)
    print(out[1].shape)
         

     
      