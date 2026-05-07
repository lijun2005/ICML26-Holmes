import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from easydict import EasyDict as edict
import math

class clip_nce(nn.Module):
    def __init__(self, reduction='mean'):
        super(clip_nce, self).__init__()
        self.reduction = reduction

    def forward(self, labels, label_dict, q2ctx_scores=None, final_labels=None):

        query_bsz = q2ctx_scores.shape[0]
        vid_bsz = q2ctx_scores.shape[1]
        diagnoal = torch.arange(query_bsz).to(q2ctx_scores.device)
        t2v_nominator = q2ctx_scores[diagnoal, labels]

        if final_labels is None:
            t2v_nominator = torch.logsumexp(t2v_nominator.unsqueeze(1), dim=1)
            t2v_denominator = torch.logsumexp(q2ctx_scores, dim=1)
            losst2v = torch.mean(t2v_denominator - t2v_nominator)
        else:
            losst2v=F.cross_entropy(q2ctx_scores,final_labels,reduction="mean")

        v2t_nominator = torch.zeros(vid_bsz).to(q2ctx_scores)
        v2t_denominator = torch.zeros(vid_bsz).to(q2ctx_scores)

        for i, label in label_dict.items():
            v2t_nominator[i] = torch.logsumexp(q2ctx_scores[label, i], dim=0)

            v2t_denominator[i] = torch.logsumexp(q2ctx_scores[:, i], dim=0)

        return losst2v + torch.mean(v2t_denominator - v2t_nominator)

class EDL_loss(nn.Module):
    def __init__(self,config):
        super(EDL_loss, self).__init__()
        self.tau = config['edl_tau']

    def mse_loss(self,labels,alpha_i2v):
        S = torch.sum(alpha_i2v, dim=1, keepdim=True)
        E = alpha_i2v - 1
        m = alpha_i2v / S
        A = torch.sum((labels - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha_i2v * (S - alpha_i2v) / (S * S * (S + 1)), dim=1, keepdim=True)
        return (A + B)



    def get_alpha(self,q2ctx_scores):
        evidences = torch.exp(torch.tanh(q2ctx_scores) / self.tau)
        alpha_i2v = evidences + 1
        return alpha_i2v
    def forward(self,labels,q2ctx_scores=None,evidences=None):
        """
        param: labels: 需要归一化
        """
        if  evidences is None:
            evidences = torch.exp(torch.tanh(q2ctx_scores) / self.tau)
        alpha_i2v = evidences + 1
        return torch.mean(self.mse_loss(labels,alpha_i2v)),alpha_i2v



class MaskedEDLLoss(nn.Module):
    def __init__(self, config):
        super(MaskedEDLLoss, self).__init__()
        self.tau = config['edl_intra_video_tau']

    def KL(self, alpha, c):
        beta = torch.ones((1, c), device=alpha.device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def mse_loss(self,labels,alpha_i2v):
        S = torch.sum(alpha_i2v, dim=1, keepdim=True)
        E = alpha_i2v - 1
        m = alpha_i2v / S
        A = torch.sum((labels - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha_i2v * (S - alpha_i2v) / (S * S * (S + 1)), dim=1, keepdim=True)
        return (A + B)



    def forward(self, sim_scores, pseudo_labels, mask):


        evidences = torch.exp(torch.tanh(sim_scores) / self.tau)
        alpha = evidences + 1


        batch_size, query_len, clip_num = alpha.shape
        alpha_reshaped = alpha.reshape(-1, clip_num)
        labels_reshaped = pseudo_labels.reshape(-1, clip_num)


        per_query_loss =  self.mse_loss(labels_reshaped,alpha_reshaped)



        per_query_loss = per_query_loss.reshape(batch_size, query_len)


        masked_loss = per_query_loss * mask


        sum_loss_per_video = torch.sum(masked_loss, dim=1)
        num_valid_queries = torch.sum(mask, dim=1).clamp(min=1e-8) # 防止除以零
        mean_loss_per_video = sum_loss_per_video / num_valid_queries


        final_loss = torch.mean(mean_loss_per_video)

        return final_loss

def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum

def DS_Combin_two(alpha1, alpha2, K):
    """
    :param alpha1: Dirichlet distribution parameters of view 1
    :param alpha2: Dirichlet distribution parameters of view 2
    :return: Combined Dirichlet distribution parameters
    """
    alpha = dict()
    alpha[0], alpha[1] = alpha1, alpha2
    b, S, E, u = dict(), dict(), dict(), dict()
    for v in range(2):
        S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
        E[v] = alpha[v] - 1
        b[v] = E[v] / (S[v].expand(E[v].shape))
        u[v] = K / S[v]

    # b^0 @ b^(0+1)
    bb = torch.bmm(b[0].view(-1, K, 1), b[1].view(-1, 1, K))
    # b^0 * u^1
    uv1_expand = u[1].expand(b[0].shape)
    bu = torch.mul(b[0], uv1_expand)
    # b^1 * u^0
    uv_expand = u[0].expand(b[0].shape)
    ub = torch.mul(b[1], uv_expand)
    # calculate C
    bb_sum = torch.sum(bb, dim=(1, 2), out=None)
    bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
    C = bb_sum - bb_diag

    # calculate b^a
    b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
    # calculate u^a
    u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

    # calculate new S
    S_a = K / u_a
    # calculate new e_k
    e_a = torch.mul(b_a, S_a.expand(b_a.shape))
    alpha_a = e_a + 1
    return e_a,alpha_a, b_a, u_a





class query_diverse_loss(nn.Module):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        self.mrg = config['neg_factor'][0]
        self.alpha = config['neg_factor'][1]
        self.lamda = config['neg_factor'][2]
        
    def forward(self, x, label_dict):

        bs = x.shape[0]
        x = F.normalize(x, dim=-1)
        cos = torch.matmul(x, x.t())

        N_one_hot = torch.zeros((bs, bs))
        for i, label in label_dict.items():
            N_one_hot[label[0]:(label[-1]+1), label[0]:(label[-1]+1)] = torch.ones((len(label), len(label)))
        N_one_hot = N_one_hot - torch.eye(bs)
        N_one_hot = N_one_hot.cuda()
    
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp))
        focal = torch.where(N_one_hot == 1, cos, torch.zeros_like(cos))
    
        neg_term = (((1 + focal) ** self.lamda) * torch.log(1 + N_sim_sum)).sum(dim=0).sum() / bs
        
        return neg_term


class loss(nn.Module):
    def __init__(self, cfg):
        super(loss, self).__init__()
        self.cfg = cfg
        self.clip_nce_criterion = clip_nce(reduction='mean')
        self.video_nce_criterion = clip_nce(reduction='mean')

        self.qdl = query_diverse_loss(cfg)



        self.intra_video_edl_loss = MaskedEDLLoss(cfg)
        self.frame_eld_loss = EDL_loss(cfg)
        self.clip_eld_loss = EDL_loss(cfg)
        self.combine_eld_loss = EDL_loss(cfg)

    def forward(self, input_list, batch):
        '''
        param: query_labels: List[int]
        param: clip_scale_scores.shape = [5*bs,bs]
        param: frame_scale_scores.shape = [5*bs,5*bs]
        param: clip_scale_scores_.shape = [5*bs,bs]
        param: frame_scale_scores_.shape = [5*bs,5*bs]
        param: label_dict: Dict[List]
        '''

        query_labels = batch['text_labels']

        clip_scale_scores = input_list[0]
        clip_scale_scores_ = input_list[1]
        label_dict = input_list[2]
        frame_scale_scores = input_list[3]
        frame_scale_scores_ = input_list[4]

        query = input_list[5]
        batched_intravideo_sim, batch_pot_results, batched_text_mask = input_list[6], input_list[7], input_list[8]
        if self.cfg['warmup_end']:
            intra_video_edl_loss = self.cfg['loss_factor'][3]*self.intra_video_edl_loss(batched_intravideo_sim, batch_pot_results, batched_text_mask)
        else:
            intra_video_edl_loss = 0.0

        assert frame_scale_scores.shape[1] == clip_scale_scores.shape[1]


        query_labels_tensor = torch.tensor(query_labels, device=frame_scale_scores.device)
        one_hot_labels = F.one_hot(query_labels_tensor, num_classes=frame_scale_scores.shape[1]).float()
        frame_trip_loss = self.get_clip_triplet_loss(frame_scale_scores, query_labels)  
        clip_trip_loss = self.get_clip_triplet_loss(clip_scale_scores, query_labels)

        if not self.cfg['warmup_end']:
            final_labels = None

            frame_eld_loss,frame_alpha = self.frame_eld_loss(one_hot_labels, frame_scale_scores)
            clip_edl_loss,clip_alpha  = self.clip_eld_loss(one_hot_labels, clip_scale_scores)
            all_evidence,all_alpha,all_b,all_u = DS_Combin_two(frame_alpha,clip_alpha,frame_scale_scores.shape[1])
            combine_edl_loss,_ = self.combine_eld_loss(labels = one_hot_labels, evidences = all_evidence)
            final_labels=None
        else:

            final_labels = one_hot_labels.clone()
            frame_alpha = self.frame_eld_loss.get_alpha(frame_scale_scores)
            clip_alpha  = self.clip_eld_loss.get_alpha( clip_scale_scores)

            frame_s_c, frame_s_n, frame_s_u = self.pair_division(frame_scale_scores, query_labels_tensor, frame_alpha, one_hot_labels)
            clip_s_c, clip_s_n, clip_s_u = self.pair_division(clip_scale_scores, query_labels_tensor, clip_alpha, one_hot_labels)


            num_queries = frame_scale_scores.shape[0]
            device = frame_scale_scores.device
            frame_s_c_mask = torch.zeros(num_queries, dtype=torch.bool, device=device); frame_s_c_mask[frame_s_c] = True
            frame_s_n_mask = torch.zeros(num_queries, dtype=torch.bool, device=device); frame_s_n_mask[frame_s_n] = True
            clip_s_c_mask = torch.zeros(num_queries, dtype=torch.bool, device=device); clip_s_c_mask[clip_s_c] = True
            clip_s_n_mask = torch.zeros(num_queries, dtype=torch.bool, device=device); clip_s_n_mask[clip_s_n] = True
            
            intersection_1 = frame_s_c_mask & clip_s_n_mask
            intersection_2 = frame_s_n_mask & clip_s_c_mask
            intersection_3 = frame_s_n_mask & clip_s_n_mask
            final_union_mask = intersection_1 | intersection_2 | intersection_3
            ambiguous_queries_set = torch.where(final_union_mask)[0]

            if ambiguous_queries_set.numel() > 0:
                beta = self.cfg['smooth_beta']
                

                p_frame = F.softmax(frame_scale_scores[ambiguous_queries_set], dim=1).detach()
                p_clip = F.softmax(clip_scale_scores[ambiguous_queries_set], dim=1).detach()
                

                ambiguous_one_hot = one_hot_labels[ambiguous_queries_set]

                smoothed_part = 0.5 * (((1 - beta) * ambiguous_one_hot + beta*p_frame) 
                                       + ((1 - beta)*ambiguous_one_hot + beta*p_clip))
                

                final_labels[ambiguous_queries_set] = smoothed_part


            frame_eld_loss, _ = self.frame_eld_loss(final_labels, evidences=frame_alpha-1)
            clip_edl_loss, _ = self.clip_eld_loss(final_labels, evidences=clip_alpha-1)
            all_evidence,all_alpha,all_b,all_u = DS_Combin_two(frame_alpha,clip_alpha,frame_scale_scores.shape[1])
            combine_edl_loss,_ = self.combine_eld_loss(labels = final_labels, evidences = all_evidence)
        all_edl_loss = (frame_eld_loss+clip_edl_loss+combine_edl_loss)*self.cfg['loss_factor'][4]

        clip_nce_loss = self.cfg['loss_factor'][0] * self.clip_nce_criterion(query_labels, label_dict, clip_scale_scores_)
        frame_nce_loss = self.cfg['loss_factor'][1] * self.video_nce_criterion(query_labels, label_dict, frame_scale_scores_)
        evidences=[frame_alpha-1,clip_alpha-1]
        consistent_dc_loss = get_dc_loss(evidences,frame_scale_scores.device)*self.cfg['loss_factor'][5] 


        qdl_loss = self.cfg['loss_factor'][2] * self.qdl(query, label_dict)

        loss_raw = clip_nce_loss + clip_trip_loss + frame_nce_loss + frame_trip_loss + qdl_loss

        loss = loss_raw+ intra_video_edl_loss + all_edl_loss+consistent_dc_loss
        return loss


    def get_clip_triplet_loss(self, query_context_scores, labels):
        v2t_scores = query_context_scores.t()
        t2v_scores = query_context_scores
        labels = np.array(labels)

        # cal_v2t_loss
        v2t_loss = 0
        for i in range(v2t_scores.shape[0]):
            pos_pair_scores = torch.mean(v2t_scores[i][np.where(labels == i)])


            neg_pair_scores, _ = torch.sort(v2t_scores[i][np.where(labels != i)[0]], descending=True)
            if self.cfg['use_hard_negative']:
                sample_neg_pair_scores = neg_pair_scores[0]
            else:
                v2t_sample_max_idx = neg_pair_scores.shape[0]
                sample_neg_pair_scores = neg_pair_scores[
                    torch.randint(0, v2t_sample_max_idx, size=(1,)).to(v2t_scores.device)]

            v2t_loss += (self.cfg['margin'] + sample_neg_pair_scores - pos_pair_scores).clamp(min=0).sum()

        # cal_t2v_loss
        text_indices = torch.arange(t2v_scores.shape[0]).to(t2v_scores.device)
        t2v_pos_scores = t2v_scores[text_indices, labels]
        mask_score = copy.deepcopy(t2v_scores.data)
        mask_score[text_indices, labels] = 999
        _, sorted_scores_indices = torch.sort(mask_score, descending=True, dim=1)
        t2v_sample_max_idx = min(1 + self.cfg['hard_pool_size'],
                                 t2v_scores.shape[1]) if self.cfg['use_hard_negative'] else t2v_scores.shape[1]
        sample_indices = sorted_scores_indices[
            text_indices, torch.randint(1, t2v_sample_max_idx, size=(t2v_scores.shape[0],)).to(t2v_scores.device)]

        t2v_neg_scores = t2v_scores[text_indices, sample_indices]

        t2v_loss = (self.cfg['margin'] + t2v_neg_scores - t2v_pos_scores).clamp(min=0)

        return t2v_loss.sum() / len(t2v_scores) + v2t_loss / len(v2t_scores)

    def pair_division(self,frame_scale_scores,query_labels_tensor,frame_alpha,one_hot_labels):

        predicted_labels = torch.argmax(frame_scale_scores, dim=1)
        correct_mask = (predicted_labels == query_labels_tensor)
        correctly_predicted_queries = torch.where(correct_mask)[0]


        consens_all_queries = torch.sum(frame_scale_scores * one_hot_labels, dim=1)


        if correctly_predicted_queries.numel() > 0:

            correct_alphas = frame_alpha[correctly_predicted_queries]
            S_correct = torch.sum(correct_alphas, dim=1)
            num_classes = frame_alpha.shape[1]
            uncertainty_correct = num_classes / S_correct
            max_uncertainty = torch.max(uncertainty_correct)
            


            beta_tensor = torch.tensor(self.cfg['division_beta'], device=frame_scale_scores.device)
            max_uncertainty = torch.min(max_uncertainty, 1 - beta_tensor)

            consens_correct = consens_all_queries[correctly_predicted_queries]

            min_consens = torch.min(consens_correct)
            min_consens = torch.max(min_consens, beta_tensor) 
        else:
            max_uncertainty = torch.tensor(1 - self.cfg['division_beta'], device=frame_scale_scores.device)
            min_consens = torch.tensor(self.cfg['division_beta'], device=frame_scale_scores.device) # consens 最大为1

        S_all = torch.sum(frame_alpha, dim=1)
        all_uncertainty = frame_alpha.shape[1] / S_all
        


        query_indices = torch.arange(frame_alpha.shape[0], device=frame_alpha.device)


        mask_U = all_uncertainty > max_uncertainty
        S_U = query_indices[mask_U]

        mask_certain = ~mask_U


        mask_C = mask_certain & (consens_all_queries >= min_consens)
        S_C = query_indices[mask_C]


        mask_n = mask_certain & (consens_all_queries < min_consens)
        S_n = query_indices[mask_n]


        S_all_entropy = torch.sum(frame_alpha, dim=1, keepdim=True)
        pred_probs = frame_alpha / S_all_entropy
        term_S = torch.digamma(S_all_entropy + 1)
        term_alpha = torch.digamma(frame_alpha + 1)

        expected_entropy = torch.sum(pred_probs * (term_S - term_alpha), dim=1)


        median_entropy = torch.median(expected_entropy)


        if S_C.numel() > 0:

            entropy_in_SC = expected_entropy[S_C]
            high_entropy_mask_in_SC = entropy_in_SC > median_entropy


            S_m = S_C[high_entropy_mask_in_SC]


            S_C = S_C[~high_entropy_mask_in_SC]
            

        else:

            S_m = torch.tensor([], dtype=torch.long, device=frame_alpha.device)


        Final_S_n = torch.cat((S_n, S_m))
        return S_C, Final_S_n, S_U