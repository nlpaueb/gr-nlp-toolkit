from torch.nn import LeakyReLU

from gr_nlp_toolkit.models.util import create_mask_from_length

from torch import nn
import torch


class DPModel(nn.Module):
    """
    Dependency Parsing model.

    Attributes:
        numrels (int): Number of dependency relation labels.
        _bert_model (nn.Module): The BERT model.
        _dp (nn.Dropout): Dropout layer.
        arc_head (nn.Linear): Linear layer for arc head representation.
        arc_dep (nn.Linear): Linear layer for arc dependent representation.
        rel_head (nn.Linear): Linear layer for relation head representation.
        rel_dep (nn.Linear): Linear layer for relation dependent representation.
        arc_bias (nn.Parameter): Bias parameter for arc representation.
        rel_bias (nn.Parameter): Bias parameter for relation representation.
        u_rel (nn.Parameter): Parameter for relation representation.
        w_arc (nn.Parameter): Parameter for arc representation.
        w_rel_head (nn.Parameter): Parameter for relation head representation.
        w_rel_dep (nn.Parameter): Parameter for relation dependent representation.
        deprel_linear_2 (nn.Linear): Linear layer for dependency relation labels.
        relu (LeakyReLU): LeakyReLU activation function.
    """

    def __init__(self, bert_model, deprel_i2l, dp):
        """
        Initialize the DPModel.

        Args:
            bert_model (nn.Module): The BERT model.
            deprel_i2l (list): List of dependency relation labels.
            dp (float): The dropout probability.

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

        self.u_rel = nn.Parameter(torch.zeros(1, 768, self.numrels * 768))

        self.w_arc = nn.Parameter(torch.zeros(1, 768, 768))
        self.w_rel_head = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))
        self.w_rel_dep = nn.Parameter(torch.zeros(1, 1, 768, self.numrels))

        self.deprel_linear_2 = nn.Linear(768, len(deprel_i2l) * 768)

        self.relu = LeakyReLU(1)


    def forward(self, text, text_len):
        """
        Forward pass of the DPModel.

        Args:
            text (Tensor): Input text.
            text_len (Tensor): Length of the input text.

        Returns:
            output (dict): Dictionary containing the output of the model.

        """
        output = {}

        attention_mask = create_mask_from_length(text_len, text.shape[1])
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