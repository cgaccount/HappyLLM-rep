import torch.nn as nn
import torch
import math

#构建RMSNorm
class RMSNorm(nn.Module):
    def __init__(self,eps,dim):
        self.eps=eps
        self.weight=nn.Parameter(torch.one(dim))

    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        return self.weight*self._norm(x.float()).type_as(x)
    
##实现旋转位置嵌入
#首先实现基函数
def precompute_freqs_cis(dim:int,end:int,theta:float=10000.0):
    '''
    dim:隐藏维度长度
    end:输入序列长度
    '''
    frqs=1/(theta**(torch.arrange(0,dim,2)[:dim//2].float()/dim))

    t=torch.arrange(0,end,device=frqs.device).float()

    frqs=torch.outer(t,frqs).float()

    frqs_sin=torch.sin(frqs)

    frqs_cos=torch.cos(frqs)

    return frqs_cos,frqs_sin

def reshape_for_broadcast(frqs_cis:torch.Tensor,x:torch.Tensor):
    ndim=x.ndim

    assert frqs_cis.shape==(x.shape[1],x.shape[-1])

    shape=[d if i==1 or i==ndim-1 else 1 for i,d in enumerate(x.shape)]

    return frqs_cis.view(shape)


#实现在q，k中进行旋转位置嵌入
def apply_ratary_emb(
        xq:torch.Tensor,
        xk:torch.Tensor,
        frqs_cos:torch.Tensor,
        frqs_sin:torch.Tensor):
    #将隐藏维度一分为二  unbind(-1)沿最后一个维度拆分
    #b,len,head_num,c//2
    xq_r,xq_i=xq.float().reshape(xq.shape[:-1]+(-1,2)).unbind(-1)
    xk_r,xk_i=xk.float().reshape(xk.shape[:-1]+(-1,2)).unbind(-1)

    #将frqs_cos和frqs_sin进行扩张，以进行广播运算
    frqs_cos=reshape_for_broadcast(frqs_cos,xq_r)
    frqs_sin=reshape_for_broadcast(frqs_sin,xq_r)

    #应用旋转，计算旋转后的实部和虚部
    xq_out_r=xq_r*frqs_cos-xq_i*frqs_sin
    xq_out_i=xq_r*frqs_sin+xq_i*frqs_cos
    xk_out_r=xk_r*frqs_cos-xk_i*frqs_sin
    xk_out_i=xk_r*frqs_sin+xk_i*frqs_cos

    xq_out=torch.stack([xq_out_r,xq_out_i],dim=-1).flatten(3)
    xk_out=torch.stack([xk_out_r,xk_out_i],dim=-1).flatten(3)

    return xq_out,xk_out

#实现分组计算注意力权重，在自回归中需要缓存所有历史token的kv，分组注意力权重计算能减少kv占用、
def repeat_kv(x_kv:torch.Tensor,n_rep:int):
    bs,l,head_num,dim=x_kv.shape

    if n_rep==1:return x_kv

    return (x_kv[:,:,:,None,:]
            .expand(bs,l,head_num,n_rep,dim)
            .reshape(bs,l,head_num*n_rep,dim))

class Attention(nn.Module):
    def __init__(self,arg):
        self.head_kv_num=arg.head_num if arg.head_kv_num is None else arg.head_kv_num
        assert arg.head_num%self.head_kv_num==0

        #默认模型单gpu运行，后续可以进行更改
        model_parallel_size=1
        self.local_head_num=arg.head_num//model_parallel_size
        self.local_head_kv_num=self.head_kv_num//model_parallel_size
        self.n_rep=self.local_head_num//self.local_head_kv_num


        self.head_dim=arg.dim//arg.head_num
        self.wq=nn.Linear(arg.dim,arg.head_num*self.head_dim,bais=False)
        self.wk=nn.Linear(arg.dim,self.head_kv_num*self.head_dim,bais=False)
        self.wv=nn.Linear(arg.dim,self.head_kv_num*self.head_dim,bais=False)

        self.flash=hasattr(torch.nn.fuctional,'scaled_dot_product_attention')
        if not self.flash:
            print("create mask")
            mask=torch.full((1,1,arg.max_seq_len,arg.max_seq_len),float("-inf"))
            mask=torch.triu(mask,diagonal=1)
            self.register_buffer("mask",mask)
            self.attention_drop=nn.Dropout(arg.dropout)

        self.droupout=arg.drpoout
        self.out=nn.Linear(self.head_dim*arg.head_num,arg.dim,bais=False)
        self.out_drop=nn.Dropout(arg.dropout)
    
    def forward(self,x,frqs_cos,frqs_sin):
        '''
        x:[bs,l,dim]
        frqs_cos:[l,dim//2]
        frqs_sin:[l,dim//2]
        frqs_cos和frqs_sin由precompute_frqs_cis获得
        '''
        bs,l,_=x.shape
        xq,xk,xv=self.wq(x),self.wk(x),self.wv(x)

        xq=xq.view(bs,l,self.local_head_num,self.head_dim)
        xk=xk.view(bs,l,self.local_head_kv_num,self.head_dim)
        xv=xv.view(bs,l,self.local_head_kv_num,self.head_dim)

        xq,xk=apply_ratary_emb(xq=xq,xk=xk,frqs_cos=frqs_cos,frqs_sin=frqs_sin)

        xk=repeat_kv(xk,self.n_rep)
        xv=repeat_kv(xv,self.n_rep)

        xq=xq.transpose(1,2)
        xk=xk.transpose(1,2)
        xv=xv.transpose(1,2)

        if self.flash:
            #is_causal是否因果遮掩
            output=torch.nn.fuctional.scaled_dot_product_attention(
                xq,xk,xv,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0 is_causal=True)
                
        else:
            score=torch.matmul(xq,xk.transpose(2,3))/math.sqrt(self.head_dim)
            assert hasattr(self,'mask')
            score=score+self.mask[:,:,:l,:l]
            score=F.softmax(score.float(),dim=-1).type_as(xq)
            score=self.attention_drop(score)
            output=torch.matmul(score,xv)

        output=output.transpose(1,2).contiguous().view(bs,l,-1)

        output=self.out(output)
        output=self.out_drop(output)

        return output












