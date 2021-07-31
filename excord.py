import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import masked_log_softmax, masked_softmax


def consistency_reg(args,
                    inputs, logits, model,
                    query_end
                    ):

    unsup_criterion = nn.KLDivLoss(reduction='none')
        
    start_logits, end_logits, class_logits = logits

    qt_mask = torch.zeros(start_logits.size())
    for i, q_end in enumerate(query_end):
        qt_mask[i, q_end:] = 1
    qt_mask = qt_mask.type(torch.float32).to(args.device)
    
    # softmax temperature controlling
    excord_softmax_temp = torch.tensor(args.excord_softmax_temp) if args.excord_softmax_temp > 0 else torch.tensor(1.)
    excord_softmax_temp = excord_softmax_temp.type(torch.float32).to(args.device)
    
    outputs = model(**inputs)

    with torch.no_grad():
        rewr_start_logits, rewr_end_logits, rewr_class_logits = outputs[1:]
        
        rewr_start_prob   = masked_softmax(rewr_start_logits  / excord_softmax_temp, qt_mask, dim=-1)
        rewr_end_prob     = masked_softmax(rewr_end_logits / excord_softmax_temp, qt_mask, dim=-1)
        rewr_label_prob   = torch.sigmoid(rewr_class_logits)
        rewr_nonlab_prob  = torch.ones(rewr_label_prob.size(), dtype=torch.float32).to(args.device) - rewr_label_prob
        rewr_class_prob   = torch.cat((rewr_label_prob, rewr_nonlab_prob), dim=-1)

    log_start_prob   = masked_log_softmax(start_logits, qt_mask, dim=-1)
    log_end_prob     = masked_log_softmax(end_logits , qt_mask, dim=-1)
    label_prob       = torch.sigmoid(class_logits)
    nonlab_prob      = torch.ones(label_prob.size(), dtype=torch.float32).to(args.device) - label_prob
    log_class_prob   = torch.log(torch.cat((label_prob, nonlab_prob), dim=-1))

    """
    Confidence-based Masking
        - loss_mask.size() = [#(Batch)]
    """ 
    if args.excord_conf_thres_cls > 0:  
        cls_loss_mask   = torch.max(rewr_class_prob, dim=-1)[0] > args.excord_conf_thres_cls
    else:
        cls_loss_mask   = torch.ones(len(class_logits))

    if args.excord_conf_thres_span > 0:
        start_loss_mask = torch.max(rewr_start_prob, dim=-1)[0] > args.excord_conf_thres_span
        end_loss_mask   = torch.max(rewr_end_prob, dim=-1)[0]   > args.excord_conf_thres_span
    else:
        start_loss_mask = torch.ones(len(class_logits))
        end_loss_mask   = torch.ones(len(class_logits))
        
    start_loss_mask = start_loss_mask.type(torch.float32).to(args.device)
    end_loss_mask   = end_loss_mask.type(torch.float32).to(args.device)
    cls_loss_mask   = cls_loss_mask.type(torch.float32).to(args.device)

    unsup_start_loss  = torch.sum(unsup_criterion(log_start_prob, rewr_start_prob), dim=-1)
    unsup_end_loss    = torch.sum(unsup_criterion(log_end_prob,   rewr_end_prob), dim=-1)
    unsup_cls_loss    = torch.sum(unsup_criterion(log_class_prob, rewr_class_prob), dim=-1)
    unsup_loss        = torch.sum(unsup_start_loss * start_loss_mask, dim=-1) / torch.max(torch.sum(start_loss_mask, dim=-1, dtype=torch.float32), torch.tensor(1.).to(args.device)) + \
                        torch.sum(unsup_end_loss * end_loss_mask, dim=-1)     / torch.max(torch.sum(end_loss_mask, dim=-1, dtype=torch.float32), torch.tensor(1.).to(args.device)) + \
                        torch.sum(unsup_cls_loss * cls_loss_mask, dim=-1)     / torch.max(torch.sum(cls_loss_mask, dim=-1, dtype=torch.float32), torch.tensor(1.).to(args.device))

    return outputs[0], unsup_loss