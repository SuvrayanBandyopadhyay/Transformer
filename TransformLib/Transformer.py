#A pytorch implementation of a transformer
import torch
import torch.nn as nn
import math

#A class to handle inp_embeddings
class inpEmbeddings(nn.Module):
    #Initialize
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    #Forward propagation
    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model) #math.sqrt(self.d_model) is used to decerase the effect of positional encodings
    
#A class to apply positional encodings to our inp embeddings
class PositionalEncodings(nn.Module):
    def __init__(self,d_model:int,seq_len:int):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        self.pe = torch.zeros(seq_len,d_model)

        #Positions
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) #We increase the dimension to get (seq_lenth,1)
        #Division term (for 2i/d model)
        div_term = torch.exp(torch.arange(0,d_model,2)).float() * -(math.log(10000.0)/d_model)

        #Now we apply sine to even positions
        self.pe[:, 0::2] = torch.sin(position*div_term)
        #And cos to odd positions
        self.pe[:, 1::2] = torch.cos(position*div_term)

        #Add another dimension to get (1,seq_len,d_model)
        self.pe = self.pe.unsqueeze(0) # (1,seq_len,d_model)

    #Forward propagation
    def forward(self,x):
        #Add positional encodings
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False) #Return (batch,seq_len,d_model)
        return x        
    
#We also normalize each layer to improve training
#This stabilizes the learning process
class LayerNorm(nn.Module):
    def __init__(self,features:int,eps:float= 10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True) #Keepdim adds extra dimension to maintain original number of dimensions

        std = x.std(dim=-1,keepdim=True)

        return self.alpha*(x-mean)/(std+self.eps) + self.bias
    


#Feed Forward Black
#Process: Apply Linear transform 
#Apply RELU
#Apply another Linear transform back to d_model

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int):
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff)
        self.linear2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear2(torch.relu(self.linear1(x)))
    

#MultiHeadAttentionBlock
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model:int, h:int):
        super().__init__()
        self.d_model = d_model
        self.h = h #Number of heads

        #Dimesion of model should be divisible by number of heads
        assert d_model%h ==0, "d_model is not divisible by h"

        self.d_k = d_model // h #Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model,d_model,bias=False) #Wq
        self.w_k = nn.Linear(d_model,d_model,bias=False) #Wk
        self.w_v = nn.Linear(d_model,d_model,bias=False) #Wv
        self.w_o = nn.Linear(d_model,d_model,bias=False) #Wo (To combine all heads)

    #We create a static method to compute attention
    @staticmethod
    def attention(query,key,value,mask):
        d_k = query.shape[-1] #Get the dimensio which we will later use 

        #Get attention squares by taking the transpose of the key
         # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)

        #If we are masking our outputs
        if mask is not None:
            #We want to repeat mask for each head
            mask = mask.unsqueeze(1).repeat(1,query.size(1),1,1)
            attention_scores.masked_fill_(mask==0,-1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        return (attention_scores@value),attention_scores
    
    #Forward propagation
    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        #We know d_model(embedding) = self.h * self.d_k
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)#We transpose it to get (batch,h,seq_len,d_k)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        #Calculate attention
        x,self.attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask)

        #We convert the form of the tensor back to the original dimensions
        # (batch, h, seq_len, d_k) => (batch, seq_len, h, d_k)
        x = x.transpose(1,2).contiguous()

        #Reshape it back to (batch, seq_len, d_model)
        x = x.view(x.shape[0],-1,self.h*self.d_k)

        #Multiply by Wo
        return self.w_o(x)

#A residual connection
class ResidualConnection(nn.Module):
    def __init__(self,features:int)->None:
        super().__init__()
        self.norm = LayerNorm(features)
    
    def forward(self,x,sublayer):
        return x + sublayer(self.norm(x))

#Encoder blocks
class EncoderBlock(nn.Module):
    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        
        self.residual_attn = ResidualConnection(features)
        self.residual_ff = ResidualConnection(features)

    def forward(self,x,src_mask):
        x = self.residual_attn(x,lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_ff(x,self.feed_forward_block)
        return x
    

#Now we finally define our encoder class
class Encoder(nn.Module):
    def __init__(self,vocab_size:int,
                 d_model:int,
                 n_layers:int,
                 h:int,
                 d_ff:int,
                 max_seq_len:int=512):
        
        super().__init__()
        self.inp_embedding = inpEmbeddings(d_model,vocab_size)
        self.positional_encoding = PositionalEncodings(d_model,max_seq_len)

        self.layers = nn.ModuleList([
            
            EncoderBlock(
                features  = d_model,
                self_attention_block=MultiHeadAttentionBlock(d_model,h),
                feed_forward_block=FeedForwardBlock(d_model,d_ff),
            ) for i in range(n_layers)
        ])
        self.norm = LayerNorm(d_model)
        
    
    def forward(self,x,src_mask):
        x = self.inp_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x,src_mask)
        
        return self.norm(x)

#A decoder block
class DecoderBlock(nn.Module):
    def __init__(self,features:int,self_attention_block:MultiHeadAttentionBlock,cross_attention_block:MultiHeadAttentionBlock,feed_forward_block:FeedForwardBlock)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block

        self.residual_self_attn = ResidualConnection(features)
        self.residual_cross_attn = ResidualConnection(features)
        self.residual_ff = ResidualConnection(features)
    
    def forward(self,x,encoder_output,src_mask,target_mask):

        
        x = self.residual_self_attn(x,lambda x: self.self_attention_block(x,x,x,target_mask))
        
        #If src_mask is None, it is a decoder only model
        if src_mask is not None:
            x = self.residual_cross_attn(x,lambda x: self.cross_attention_block(x,encoder_output,encoder_output,src_mask))   
        
        x = self.residual_ff(x,self.feed_forward_block)
        return x    

#Now we create a decoder class
class Decoder(nn.Module):
    def __init__(self,vocab_size:int,
                 d_model:int,
                 n_layers:int,
                 h:int,
                 d_ff:int,
                 max_seq_len:int=512):
        super().__init__()
        self.inp_embedding = inpEmbeddings(d_model,vocab_size)
        self.positional_encoding = PositionalEncodings(d_model,max_seq_len)

        self.layers = nn.ModuleList([
            DecoderBlock(
                features=d_model,
                self_attention_block=MultiHeadAttentionBlock(d_model,h),
                cross_attention_block = MultiHeadAttentionBlock(d_model,h),
                feed_forward_block = FeedForwardBlock(d_model,d_ff)
            )
        for i in range(n_layers)
        ]) 

        self.norm = LayerNorm(d_model)

    #Forward propagation
    def forward(self,x,encoder_output,src_mask,target_mask):

        x = self.inp_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,target_mask)

        return self.norm(x)
