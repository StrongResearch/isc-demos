from copy import deepcopy

import clip
import open_clip
import torch
import os
from torchmetrics import (
    Metric,
    MultiScaleStructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)
import time
import torch.distributed as dist
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize


class MSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.persistent(True)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        images = images.view(images.shape[0], -1)
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        err = ((images - reconstructions) ** 2).mean(-1)

        self.error += err.sum()
        self.total += images.shape[0]

    def reduce(self):
        e = self.error.clone()
        t = self.total.clone()
        dist.all_reduce(e, op=dist.ReduceOp.SUM)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        
        return (e, t)
    
    def compute(self):
        return self.error.float() / self.total


class PSNR(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("psnr", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.persistent(True)

        self.func = PeakSignalNoiseRatio(data_range=1.0, reduction=None, dim=1)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        images = images.contiguous()
        reconstructions = reconstructions.contiguous()

        # Undo normalization and transform images to 0 to 1 range
        images = images.view(images.shape[0], -1)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = reconstructions.view(reconstructions.shape[0], -1)
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute PSNR
        psnr = self.func(reconstructions, images).sum()
        if not torch.isinf(psnr):
            self.psnr += psnr
            self.total += images.shape[0]
        else:
            print("PSNR is inf, skipping sample")

    def reduce(self):
        p = self.psnr.clone()
        t = self.total.clone()
        dist.all_reduce(p, op=dist.ReduceOp.SUM)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        
        return (p, t)
    
    def compute(self):
        return self.psnr.float() / self.total


class MS_SSIM(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("ms_ssim", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.persistent(True)

        self.func = MultiScaleStructuralSimilarityIndexMeasure(reduction="none", data_range=1.0)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)

        # Compute MS-SSIM
        ms_ssim = self.func(reconstructions, images).sum()

        self.ms_ssim += ms_ssim
        self.total += images.shape[0]

    def reduce(self):
        m = self.ms_ssim.clone()
        t = self.total.clone()
        dist.all_reduce(self.ms_ssim, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
        m_o = self.ms_ssim.clone()
        t_o = self.total.clone()
        self.error = m
        self.total = t
        
        return (m_o, t_o)
    
    def compute(self):
        return self.ms_ssim.float() / self.total


class FID_Inception(Metric):
    def __init__(self):
        super().__init__()
        self.func = FrechetInceptionDistance(normalize=True)
        self.add_state("images", [])
        self.add_state("reconstructions", [])
        self.persistent(True)
        
    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape

        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0).expand(-1, 3, -1, -1)

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0).expand(
            -1, 3, -1, -1
        )
        
        self.images.append(images.clone().to(f"cuda:{os.environ['LOCAL_RANK']}"))
        self.reconstructions.append(reconstructions.clone().to(f"cuda:{os.environ['LOCAL_RANK']}"))
        
        # Compute FID
        # self.func.update(images, real=True)
        # self.func.update(reconstructions, real=False)

    def state_dict(self):
        return {"images": self.images, "reconstructions": self.reconstructions}
    
    def load_state_dict(self, checkpoint, _):
        self.reconstructions = checkpoint["reconstructions"]
        self.images = checkpoint["images"]
        
    def reduce(self):
        ims = torch.stack(self.images)
        world_size = dist.get_world_size()
        local_size = torch.tensor(ims.size(), device=ims.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        max_length = max(size[0] for size in all_sizes)

        length_diff = max_length.item() - local_size[0].item()
        if length_diff:
            pad_size = (length_diff, *ims.size()[1:])
            padding = torch.zeros(pad_size, device=ims.device, dtype=ims.dtype)
            ims = torch.cat((ims, padding))

        all_tensors_padded = [torch.zeros_like(ims) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, ims)
        all_ims = []
        for tensor_, size in zip(all_tensors_padded, all_sizes):
            all_ims.append(tensor_[:size[0]])

        rec = torch.stack(self.reconstructions)
        local_size = torch.tensor(rec.size(), device=rec.device)
        all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(all_sizes, local_size)

        max_length = max(size[0] for size in all_sizes)

        length_diff = max_length.item() - local_size[0].item()
        if length_diff:
            pad_size = (length_diff, *rec.size()[1:])
            padding = torch.zeros(pad_size, device=rec.device, dtype=rec.dtype)
            rec = torch.cat((rec, padding))

        all_tensors_padded = [torch.zeros_like(rec) for _ in range(world_size)]
        dist.all_gather(all_tensors_padded, rec)
        all_recs = []
        for tensor_, size in zip(all_tensors_padded, all_sizes):
            all_recs.append(tensor_[:size[0]])
        
        return (all_ims, all_recs)
    
    
    def compute(self):
        all_ims, all_recs = self.reduce()
        for item in all_recs[0]:
            self.func.update(item.to(f"cuda:{os.environ['LOCAL_RANK']}"), real=False)
        for item in all_ims[0]:
            self.func.update(item.to(f"cuda:{os.environ['LOCAL_RANK']}"), real=True)
        
        # for item in all_recs[1]:
        #     self.func.update(item.to(f"cuda:{os.environ['LOCAL_RANK']}"), real=False)
        # for item in all_ims[1]:
        #     self.func.update(item.to(f"cuda:{os.environ['LOCAL_RANK']}"), real=True)
        # print(f"len of ims: {len(all_ims[0])} and {len(all_ims[1])}")
        return self.func.compute()


class FID_CLIP(Metric):
    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False

    real_features_sum: torch.Tensor
    real_features_cov_sum: torch.Tensor
    real_features_num_samples: torch.Tensor

    fake_features_sum: torch.Tensor
    fake_features_cov_sum: torch.Tensor
    fake_features_num_samples: torch.Tensor

    def __init__(self, version, reset_real_features=True, normalize=False, **kwargs):
        
        super().__init__(**kwargs)
        if version == "CLIP":
            self.clip, _ = clip.load("ViT-B/32", device=f"cuda:{os.environ.get('LOCAL_RANK',-1)}")
            res = self.clip.visual.input_resolution
        elif version == "BiomedCLIP":
            model_kwargs = {'hf_model_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 
                                         'hf_tokenizer_name': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract', 'hf_proj_type': 'mlp', 
                                         'hf_pooler_type': 'cls_last_hidden_state_pooler', 'context_length': 256, 'hf_model_pretrained': False}
            self.clip, _, _ = open_clip.create_model_and_transforms(
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", text_cfg=model_kwargs, device=f"cuda:{os.environ.get('LOCAL_RANK',-1)}"
            )
            res = 224
        for _, param in enumerate(self.clip.parameters()):
            param.requires_grad_(False)
        self.clip.eval()
        self.clip_transform = Compose(
            [
                Resize(res, interpolation=3),
                CenterCrop(res),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
                ),
            ]
        )

        self.reset_real_features = reset_real_features
        self.normalize = normalize

        num_features = 512
        mx_nb_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "real_features_cov_sum", torch.zeros(mx_nb_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state(
            "fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum"
        )
        self.add_state(
            "fake_features_cov_sum", torch.zeros(mx_nb_feats).double(), dist_reduce_fx="sum"
        )
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")
        self.persistent(True)

    def update(self, images, reconstructions):
        assert images.shape == reconstructions.shape
        # Undo normalization and transform reconstructions to 0 to 1 range
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images = images.expand(-1, 3, -1, -1)
        # Compute features for real images
        imgs = self.clip_transform(images)
        with torch.no_grad():
            features = self.clip.encode_image(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.real_features_sum += features.sum(dim=0)
        self.real_features_cov_sum += features.t().mm(features)
        self.real_features_num_samples += imgs.shape[0]

        # Undo normalization and transform reconstructions to 0 to 1 range
        reconstructions = torch.clamp((reconstructions + 1.0) / 2.0, min=0.0, max=1.0)
        reconstructions = reconstructions.expand(-1, 3, -1, -1)
        # Compute features for reconstructed images
        imgs = self.clip_transform(reconstructions)
        with torch.no_grad():
            features = self.clip.encode_image(imgs)
        self.orig_dtype = features.dtype
        features = features.double()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        self.fake_features_sum += features.sum(dim=0)
        self.fake_features_cov_sum += features.t().mm(features)
        self.fake_features_num_samples += imgs.shape[0]

    def compute(self):
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(
            self.orig_dtype
        )
        
    def state_dict(self, dtype=None):
        return {"real_features_sum": self.real_features_sum, "real_features_cov_sum": self.real_features_cov_sum,
                "real_features_num_samples": self.real_features_num_samples, "fake_features_sum": self.fake_features_sum,
                "fake_features_cov_sum": self.fake_features_cov_sum, "fake_features_num_samples": self.fake_features_num_samples,
                "orig_dtype": dtype}

    
    def load_state_dict(self, checkpoint, _):
        self.real_features_sum = checkpoint["real_features_sum"]
        self.real_features_cov_sum = checkpoint["real_features_cov_sum"]
        self.real_features_num_samples = checkpoint["real_features_num_samples"]
        self.fake_features_sum = checkpoint["fake_features_sum"]
        self.fake_features_cov_sum = checkpoint["fake_features_cov_sum"]
        self.fake_features_num_samples = checkpoint["fake_features_num_samples"]
        self.orig_dtype = checkpoint["orig_dtype"]
    
    def reduce(self):
        a = self.real_features_sum.clone()
        b = self.real_features_cov_sum.clone()
        c = self.real_features_num_samples.clone()
        d = self.fake_features_sum.clone()
        e = self.fake_features_cov_sum.clone()
        f = self.fake_features_num_samples.clone()
        
        dist.all_reduce(a, op=dist.ReduceOp.SUM)
        dist.all_reduce(b, op=dist.ReduceOp.SUM)
        dist.all_reduce(c, op=dist.ReduceOp.SUM)
        dist.all_reduce(d, op=dist.ReduceOp.SUM)
        dist.all_reduce(e, op=dist.ReduceOp.SUM)
        dist.all_reduce(f, op=dist.ReduceOp.SUM)
        
        return (a, b, c, d, e, f)

    def reset(self) -> None:
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()
