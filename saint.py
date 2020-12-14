import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )
        

class Encoder_block(nn.Module):

    def __init__(self , dim_model, heads_en, total_ex ,total_cat):
        super().__init__()

        # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_ex =   nn.Embedding( total_ex , embedding_dim = dim_model )
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_model )
        #embd_pos =  nn.Embedding( total_pos, embedding_dim = dim_model )

        # multihead attention
        self.multi_en = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_en,  )       ## todo add dropout, LayerNORM

        # feedforward block 
        self.ffn_en = Feed_Forward_block( dim_model )                                              ## todo dropout, LayerNorm


    def forward(self, in_ex, in_cat, first_block=True):

        ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_cat = self.embd_cat( in_cat )
            #in_pos = self.embd_pos( in_pos )

            #combining the embedings
            out = in_ex + in_cat #+ in_pos         # (b,n,d)
        else:
            out = in_ex
        
        out = out.permute(1,0,2)               # (n,b,d)  # print('pre multi', out.shape )
        #Multihead attention                            ## todo verify if E to passed as q,k,v
        out, attn_wt = self.multi_en( out , out , out ) ## todo pass attention mask upper triangular

        #feed forward
        out = out.permute(1,0,2)               # (b,n,d)
        out = self.ffn_en( out )

        return out


class Decoder_block(nn.Module):

    def __init__(self,dim_model ,total_in, heads_de,  ):
        super().__init__()

        #interaction embedding
        self.embd_in = nn.Embedding(  total_in , embedding_dim = dim_model )

        # M1 multihead for interaction embedding as q k v
        self.multi_de1 = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  

        # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.multi_de2 = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_de  )  

        # feed forward layer
        self.ffn_en = Feed_Forward_block( dim_model )  

    def forward(self, in_in, en_out,first_block=True):

         ## todo create a positional encoding ( two options numeric, sine)
        if first_block:
            in_in = self.embd_in( in_in )
            #in_pos = self.embd_pos( in_pos )

            #combining the embedings
            out = in_in #+ in_cat #+ in_pos         # (b,n,d)
        else:
            out = in_in
        
        out = out.permute(1,0,2)               # (n,b,d)# print('pre multi', out.shape )

        #Multihead attention M1                                      ## todo verify if E to passed as q,k,v
        out, attn_wt = self.multi_de1( out , out , out )             ## todo pass attention mask upper triangular

        #Multihead attention M2                                      ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1,0,2)           # (b,n,d)-->(n,b,d)
        out, attn_wt = self.multi_de2( out , en_out , en_out )       ## todo pass attention mask upper triangular

        #feed forward
        out = out.permute(1,0,2)               # (b,n,d)
        out = self.ffn_en( out )                                     ## todo layernorm, skipconnection 

        return out

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class saint(nn.Module):
    def __init__(self,dim_model,num_en, num_de ,heads_en, total_ex ,total_cat,total_in,heads_de):
        super().__init__( )

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones( Encoder_block(dim_model, heads_en , total_ex ,total_cat) , num_en)
        self.decoder = get_clones( Decoder_block(dim_model ,total_in, heads_de)             , num_de)

        self.out = nn.Linear(in_features= dim_model , out_features=1)
    
    def forward(self,in_ex, in_cat,  in_in ):
        
        # pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x>=1:
                first_block = False
            in_ex = self.encoder[x]( in_ex, in_cat ,first_block=first_block)
            in_cat = in_ex                                  # passing same output as q,k,v to next encoder block

        # pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x>=1:
                first_block = False
            in_in = self.decoder[x]( in_in , en_out= in_ex, first_block=first_block )

        # Output layer
        in_in = torch.sigmoid( self.out( in_in ) )
        return in_in






seq_len = 100
total_ex = 1200
total_cat = 234
total_in = 2


def random_data(bs, seq_len , total_ex, total_cat, total_in = 2):
    ex = torch.randint( 0 , total_ex ,(bs , seq_len) )
    cat = torch.randint( 0 , total_cat ,(bs , seq_len) )
    de = torch.randint( 0 , total_in ,(bs , seq_len) )
    return ex,cat, de


in_ex, in_cat, in_de = random_data(64, seq_len , total_ex, total_cat, total_in)


model = saint(dim_model=128,
            num_en=6,
            num_de=6,
            heads_en=8,
            heads_de=8,
            total_ex=total_ex,
            total_cat=total_cat,
            total_in=total_in,
            )

outs = model(in_ex, in_cat, in_de)

print(outs.shape)
