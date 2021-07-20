from torch.nn import LeakyReLU

import pytorch_wrapper.functional as pwF

from torch import nn
import torch


class DPModel(nn.Module):

    def __init__(self, bert_model, deprel_i2l, dp):
        """
        :param bert_model:  The bert model nn.Module
        :param dp: the drop out probability
        """
        super(DPModel, self).__init__()
        self.numrels = len(deprel_i2l)
        self._bert_model = bert_model
        self._dp = nn.Dropout(dp)

        self.arc_head = nn.Linear(768, 768)
        self.arc_dep = nn.Linear(768, 768)

        self.rel_head = nn.Linear(768, 768)
        self.rel_dep = nn.Linear(768, 768)

        self.arc_bias = nn.Parameter(torch.zeros(1, 768, 1))
        self.rel_bias = nn.Parameter(torch.zeros(1, 1, 1, self.numrels))

        # todo initialization
        self.u_rel = nn.Parameter(torch.zeros(1,768, self.numrels * 768))

        self.w_arc = nn.Parameter(torch.zeros(1,768, 768))
        self.w_rel_head = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))
        self.w_rel_dep = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))

        self.deprel_linear_2 = nn.Linear(768, len(deprel_i2l) * 768)

        self.relu = LeakyReLU(1)


    def forward(self, text, text_len):
        # bs : batch size
        # mseq: maximum sentence length in tokens for the current batch
        # numrels: number of dependency relation labels
        output = {}

        attention_mask = pwF.create_mask_from_length(text_len, text.shape[1])
        bert = self._bert_model(text, attention_mask=attention_mask)

        # output size bs , mseq , 768
        bert_output = self._dp(bert[0])
        bs = bert_output.shape[0]
        mseq = bert_output.shape[1]

        # Specialized vector representations
        arc_head = self.relu(self.arc_head(bert_output))  # bs,mseq,768
        arc_dep = self.relu(self.arc_dep(bert_output))  # bs,mseq,768
        rel_head = self.relu(self.rel_head(bert_output))  # bs,mseq,768
        rel_dep = self.relu(self.rel_dep(bert_output))  # bs,mseq,768

        # bs,mseq,768 @ bs,768,mseq + bs,mseq,768 @ 1,768,1
        output_linear_head = arc_head @ (arc_dep @ self.w_arc).transpose(1, 2) + arc_head @ self.arc_bias
        # arcdep * self w.arc  = (bs,mseq,768) * (1,768,768) = (bs, mseq , 768)

        # bs,mseq, 768 * 1,768,768 *numrel = bs,mseq,numrel,768,`
        label_biaffine = rel_dep @ self.u_rel # bs,mseq,768 * numrel
        label_biaffine = label_biaffine.reshape(bs,mseq,self.numrels,768)
        label_biaffine = label_biaffine @ rel_head.transpose(1,2).unsqueeze(1) # bs,mseq,numrel,mseq
        label_biaffine = label_biaffine.transpose(2,3)

        label_head_affine = (rel_head.unsqueeze(2) @ self.w_rel_head)
        label_dep_affine = (rel_dep.unsqueeze(2) @ self.w_rel_dep)
        label_bias = self.rel_bias

        output_linear_rel = label_biaffine + label_head_affine + label_dep_affine + label_bias
        #                   (bs,mseq,1,768) @ (1 , 1 , 768 ,numrels) + ( 1, 1 , 1, numrels)
        #                    (bs,mseq,1,numrels)

        output['heads'] = output_linear_head
        output['deprels'] = output_linear_rel.reshape(bs, mseq, mseq, self.numrels)

        selected_arcs = output_linear_head.argmax(-1)  # bs,mseq (indexes in [0,mseq) )
        selected_arcs = selected_arcs.unsqueeze(-1).repeat(1, 1, mseq)  # bs,mseq,mseq
        selected_arcs = selected_arcs.unsqueeze(-1).repeat(1, 1, 1, self.numrels)  # bs,mseq,mseq, numrels

        deprels_output = torch.gather(output_linear_rel, dim=2, index=selected_arcs)  # bs,mseq,mseq,numrels
        # dim 2 is redundant so must be deleted ( there is only one head for every token)
        deprels_output = deprels_output.narrow(2, 0, 1)  # bs,mseq,1,numrels
        deprels_output = deprels_output.squeeze(2)  # bs , mseq,numrels
        output['gathered_deprels'] = deprels_output

        return output