import torch
from torch import nn


class ConModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
      
        super(ConModule, self).__init__()
       
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) 
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
       
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) 
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] 
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) 
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) 
        
     
        return pseudo_aligned_out, (pred_output_position_inclu_blank)


    