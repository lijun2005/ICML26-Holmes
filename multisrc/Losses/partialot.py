import torch
from torch import nn
import numpy as np
from torch.nn import functional as F



class POT(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.scale = config.pot_scale  
        self.prompt_ratio = config.pot_prompt_ratio
        self.sinkhorn_iterations = 50

    def forward(self, temp_clip_emb, temp_text_emb):
        clip_logit = torch.matmul(F.normalize(temp_clip_emb, dim=-1), F.normalize(temp_text_emb, dim=-1).t()).permute(1, 0)

        clip_logit = (clip_logit - torch.min(clip_logit)) / (torch.max(clip_logit) - torch.min(clip_logit))  # normalize the logits


        if self.prompt_ratio is not None:

            all_logits = clip_logit.reshape(-1)
            k = int(self.prompt_ratio * all_logits.numel())
            prompt_token_clip, _ = torch.kthvalue(all_logits, k)
        else:
            prompt_token_clip = None


        # calculate the video-paragraph similarity matrix
        logits_prompt,mass = self.calculate_video_logits(clip_logit.unsqueeze(0), prompt_token_clip)

        return logits_prompt.squeeze()

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                                sinkhorn_iterations: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(sinkhorn_iterations):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)
    
    @torch.no_grad()
    def log_optimal_transport_prompt(self, scores: torch.Tensor, sinkhorn_iterations: int, P: torch.Tensor) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability, with prompt bucket

        Args:
          scores: similarity matrix:[batchsize,text_len,video_len]
          sinkhorn_iterations: number of iterations to run sinkhorn
          P: value of prompt bucket
        Returns:
            Z: transport assignment matrix (log probabilities)
        """
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m * one).to(scores), (n * one).to(scores)

        bins1 = P.expand(b, 1, n)
        couplings = torch.cat([scores, bins1], 1)


        norm = - ns.log()
        log_nu = norm.expand(n)

        # Row marginals (mu): total mass must also be 1
        # Mass for real rows is 1/(N+M), for bucket row is N/(N+M)
        row_norm = -(ns + ms).log() # log(1/(N+M))
        
        # Real rows (M rows) get mass 1/(N+M) each
        real_rows_mu = row_norm.expand(m)
        
        # Bucket row gets mass N/(N+M)
        # log(N/(N+M)) = log(N) + log(1/(N+M))
        bucket_row_mu = ns.log() + row_norm
        
        log_mu = torch.cat([real_rows_mu, bucket_row_mu.unsqueeze(0)])


        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, sinkhorn_iterations)
        Z = Z - norm  # multiply probabilities by M+N

        # remove the prompt bucket from the OT assignment matrix
        Z = Z[:, :m, :n]
        return Z

    def calculate_video_logits(self, distance, prompt_token):
        """output logits of video-paragraph similarity matrix
        Args:
          distance: clip-caption similarity matrix
          prompt_token: value of the prompt bucket
        """
        # re-align clip with caption through optimal transport, align_logit is the transport assignment matrix
        align_logit = self.log_optimal_transport_prompt(distance / self.scale, self.sinkhorn_iterations, prompt_token / self.scale)
        # compute moving mass of optimal transport for each video-paragraph pair
        mass = align_logit.exp().sum(dim=[1, 2], keepdim=True) 

        return align_logit.exp(), mass



class BatchPOT(nn.Module):
    def __init__(self, config):
        """Constructor.
        Args:
          scale: smoothness parameter $\varepsilon$ In Eq. (2-3)
          prompt_ratio: select the bottom x% similarity of the original aligned clip-caption pairs as the value of bucket in Eq. (6)
          sinkhorn_iterations: number of iterations for Sinkhorn Normalization in Eq. (3)
        """
        super().__init__()
        self.scale = config.pot_scale  # smoothness parameter $\varepsilon$ In Eq. (2-3)
        self.prompt_ratio = config.pot_prompt_ratio
        self.sinkhorn_iterations = 50

    def forward(self, temp_clip_emb, temp_text_emb, text_mask):
        # temp_clip_emb: (B, C, D), temp_text_emb: (B, Q, D), text_mask: (B, Q)
        # B: batch_size, C: num_clips, Q: max_queries, D: feat_dim
        

        clip_logit = torch.bmm(F.normalize(temp_clip_emb, dim=-1), F.normalize(temp_text_emb, dim=-1).transpose(1, 2))


        b, c, q = clip_logit.shape
        clip_logit_flat = clip_logit.view(b, -1)
        min_vals = torch.min(clip_logit_flat, dim=1, keepdim=True)[0].unsqueeze(-1)
        max_vals = torch.max(clip_logit_flat, dim=1, keepdim=True)[0].unsqueeze(-1)
        clip_logit = (clip_logit - min_vals) / (max_vals - min_vals + 1e-8)

        if self.prompt_ratio is not None:

            prompt_token_clip_list = []
            for i in range(b):

                valid_queries_mask = text_mask[i].bool()
                valid_logits = clip_logit[i, :, valid_queries_mask].reshape(-1)
                
                if valid_logits.numel() == 0: 
                    prompt_token_clip_list.append(torch.tensor(0.0, device=clip_logit.device))
                    continue

                k = int(self.prompt_ratio * valid_logits.numel())

                k = max(1, min(k, valid_logits.numel()))
                prompt_token_clip, _ = torch.kthvalue(valid_logits, k)
                prompt_token_clip_list.append(prompt_token_clip)
            
            prompt_token_clip = torch.stack(prompt_token_clip_list) # shape: (B,)
        else:
            prompt_token_clip = None


        logits_prompt, mass = self.calculate_video_logits(clip_logit.permute(0, 2, 1), prompt_token_clip, text_mask)

        return clip_logit.permute(0, 2, 1),logits_prompt, mass

    def log_sinkhorn_iterations(self, Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor,
                                sinkhorn_iterations: int) -> torch.Tensor:
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for i in range(sinkhorn_iterations):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)

    @torch.no_grad()
    def log_optimal_transport_prompt(self, scores: torch.Tensor, sinkhorn_iterations: int, P: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        """ Perform Differentiable Optimal Transport in Log-space for stability, with prompt bucket

        Args:
          scores: similarity matrix:[batchsize, text_len, video_len]
          sinkhorn_iterations: number of iterations to run sinkhorn
          P: value of prompt bucket [batchsize]
          text_mask: [batchsize, text_len]
        Returns:
            Z: transport assignment matrix (log probabilities)
        """
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        

        ms = text_mask.sum(dim=1) # (B,)
        ns = (n * one).to(scores).expand(b) # (B,)

        bins1 = P.view(b, 1, 1).expand(b, 1, n)
        couplings = torch.cat([scores, bins1], 1)

        norm = -torch.log(ns) # (B,)
        log_nu = norm.unsqueeze(1).expand(b, n) # (B, n)

        row_norm = -torch.log(ns + ms) # (B,)
        

        real_rows_mu = row_norm.unsqueeze(1).expand(b, m)

        real_rows_mu = real_rows_mu.masked_fill(text_mask == 0, -1e9)


        bucket_row_mu = torch.log(ns) + row_norm # (B,)
        
        log_mu = torch.cat([real_rows_mu, bucket_row_mu.unsqueeze(1)], dim=1) # (B, m+1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, sinkhorn_iterations)
        Z = Z - norm.view(b, 1, 1)  # multiply probabilities by N


        Z = Z[:, :m, :n]
        return Z

    def calculate_video_logits(self, distance, prompt_token, text_mask):
        """output logits of video-paragraph similarity matrix
        Args:
          distance: clip-caption similarity matrix [B, Q, C]
          prompt_token: value of the prompt bucket [B]
          text_mask: [B, Q]
        """
        align_logit = self.log_optimal_transport_prompt(distance / self.scale, self.sinkhorn_iterations, prompt_token / self.scale, text_mask)
        

        align_logit_exp = align_logit.exp()

        align_logit_exp = align_logit_exp * text_mask.unsqueeze(-1)

        row_sums = align_logit_exp.sum(dim=2, keepdim=True) + 1e-8
        align_logit_exp = align_logit_exp / row_sums

        mass = align_logit_exp.sum(dim=[1, 2]) # (B,)

        return align_logit_exp, mass