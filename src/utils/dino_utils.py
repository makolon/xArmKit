import math
import types
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import mediapy as media
import numpy as np
import timm
import torch
import torch.nn.modules.utils as nn_utils
from PIL import Image
from sklearn.cluster import KMeans, MeanShift
from torch import nn
from torchvision import transforms

import robofin.utils.transform_utils as tru

num_pairs = 8
load_size = 480
layer = 11
facet = "key"
bin = True
thresh = 0.2
model_type = "dinov3_vits16"  # DINOv3 model with patch size 16
stride = 4  # DINOv3 vitb16 has patch size 16, so stride must be divisor of 16 (1, 2, 4, 8, or 16)
patch_size = 16  # DINOv3 vitb16 has patch size 16
device = "cuda"  # "cuda" if torch.cuda.is_available() else "cpu"


class ViTExtractor:
    """This class facilitates extraction of features, descriptors, and saliency maps from a ViT.

    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(
        self,
        model_type: str = "dinov3_vitb16",
        stride: int = 4,
        model: nn.Module = None,
        device: str = "cuda",
    ):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov3_vitg16 |
                          dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14 |
                          dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 |
                          vit_small_patch8_224 | vit_small_patch16_224 |
                          vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)

        # Handle patch_size - can be int or tuple
        patch_size = self.model.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            self.p = patch_size[0]  # Use first element if tuple
        else:
            self.p = patch_size

        self.stride = self.model.patch_embed.proj.stride

        self.mean = (
            (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )
        self.std = (
            (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        )

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load.
                           DINOv3: [dinov3_vits16 | dinov3_vitb16 | dinov3_vitl16 | dinov3_vitg16]
                           DINOv2: [dinov2_vits14 | dinov2_vitb14 | dinov2_vitl14 | dinov2_vitg14]
                           DINO v1: [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16]
                           timm: [vit_small_patch8_224 | vit_small_patch16_224 |
                                  vit_base_patch8_224 | vit_base_patch16_224]
        :return: the model
        """
        if "dinov3" in model_type:
            # DINOv3 models
            cwd_dir = Path(__file__).parent.resolve()
            repo_dir = cwd_dir / "dinov3"
            weights_dir = (
                cwd_dir
                / "dinov3"
                / "weights"
                / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
            )
            model = torch.hub.load(
                str(repo_dir),
                "dinov3_vitb16",
                source="local",
                weights=str(weights_dir),
            )
        elif "dinov2" in model_type:
            # DINOv2 models
            model = torch.hub.load("facebookresearch/dinov2", model_type)
        elif "dino" in model_type:
            # DINO v1 models
            model = torch.hub.load("facebookresearch/dino:main", model_type)
        else:
            # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
                "vit_small_patch16_224": "dino_vits16",
                "vit_small_patch8_224": "dino_vits8",
                "vit_base_patch16_224": "dino_vitb16",
                "vit_base_patch8_224": "dino_vitb8",
            }
            model = torch.hub.load(
                "facebookresearch/dino:main", model_type_dict[model_type]
            )
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict["head.weight"]
            del temp_state_dict["head.bias"]
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """

        def interpolate_pos_encoding(
            self, x: torch.Tensor, w: int, h: int
        ) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (
                w0 * h0 == npatch
            ), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(
                    1, int(math.sqrt(N)), int(math.sqrt(N)), dim
                ).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert (
                int(w0) == patch_pos_embed.shape[-2]
                and int(h0) == patch_pos_embed.shape[-1]
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size_raw = model.patch_embed.patch_size
        # Handle patch_size - can be int or tuple
        if isinstance(patch_size_raw, tuple):
            patch_size = patch_size_raw[0]  # Use first element if tuple
        else:
            patch_size = patch_size_raw

        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in stride]), (
            f"stride {stride} should divide patch_size {patch_size}"
        )

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(
            ViTExtractor._fix_pos_enc(patch_size, stride), model
        )
        return model

    def preprocess(
        self,
        image_path: Union[str, Path],
        load_size: Union[int, Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        pil_image = Image.open(image_path).convert("RGB")
        if load_size is not None:
            pil_image = transforms.Resize(
                load_size, interpolation=transforms.InterpolationMode.LANCZOS
            )(pil_image)
        prep = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.mean, std=self.std)]
        )
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ["attn", "token"]:

            def _hook(model, input, output):
                self._feats.append(output)

            return _hook

        if facet == "query":
            facet_idx = 0
        elif facet == "key":
            facet_idx = 1
        elif facet == "value":
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = (
                module.qkv(input)
                .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            self._feats.append(qkv[facet_idx])  # Bxhxtxd

        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, layers: List[int] = 11, facet: str = "key"
    ) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (
            1 + (H - self.p) // self.stride[0],
            1 + (W - self.p) // self.stride[1],
        )
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(
            B, bin_x.shape[1], self.num_patches[0], self.num_patches[1]
        )
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3**k
            avg_pool = torch.nn.AvgPool2d(
                win_size, stride=1, padding=win_size // 2, count_include_pad=False
            )
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros(
            (B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])
        ).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3**k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(
                            x - kernel_size, x + kernel_size + 1, kernel_size
                        ):
                            if i == y and j == x and k != 0:
                                continue
                            if (
                                0 <= i < self.num_patches[0]
                                and 0 <= j < self.num_patches[1]
                            ):
                                bin_x[
                                    :,
                                    part_idx * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[
                                    :,
                                    part_idx * sub_desc_dim : (part_idx + 1)
                                    * sub_desc_dim,
                                    y,
                                    x,
                                ] = avg_pools[k][:, :, temp_i, temp_j]
                            part_idx += 1
        bin_x = (
            bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        )
        # Bx1x(t-1)x(dxh)
        return bin_x

    def _n_extra_tokens(self, t: int) -> int:
        hp, wp = self.num_patches
        return max(int(t - (hp * wp)), 0)

    def extract_descriptors(
        self,
        batch: torch.Tensor,
        layer: int = 11,
        facet: str = "key",
        bin: bool = False,
        include_cls: bool = False,
    ) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in [
            "key",
            "query",
            "value",
            "token",
        ], (
            f"""{facet} is not a supported facet for descriptors. choose from ['key' | 'query' | 'value' | 'token'] """
        )
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]  # 'key'|'query'|'value': Bxhxtxd, 'token': Bxtxd
        if facet == "token":
            x.unsqueeze_(dim=1)
        tdim = 2  # Bxhxtxd
        t_all = x.shape[tdim]
        n_extra = self._n_extra_tokens(t_all)  # = CLS(1) + register(k)
        if not include_cls:
            x = x[:, :, n_extra:, :]
        else:
            keep_cls = 1
            drop_regs = max(n_extra - keep_cls, 0)
            if drop_regs > 0:
                cls_tok = x[:, :, :keep_cls, :]
                rest = x[:, :, keep_cls + drop_regs :, :]
                x = torch.cat([cls_tok, rest], dim=tdim)
        if not bin:
            desc = (
                x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
            )  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        supported_models = [
            "dino_vits8",
            "dino_vits16",
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov3_vits16",
            "dinov3_vitb16",
        ]
        assert self.model_type in supported_models, (
            f"saliency maps are supported only for {supported_models}."
        )

        num_blocks = len(self.model.blocks)
        last_layer = num_blocks - 1

        q = self._extract_features(batch, [last_layer], "query")[0]
        k = self._extract_features(batch, [last_layer], "key")[0]

        scale = q.shape[-1] ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = torch.softmax(attn, dim=-1)

        hp, wp = self.num_patches
        t_all = attn.shape[-1]
        n_extra = max(int(t_all - (hp * wp)), 0)
        start = n_extra
        cls_attn = attn[:, :, 0, start:]
        cls_attn = cls_attn.mean(dim=1)

        mins, maxs = cls_attn.min(dim=1)[0], cls_attn.max(dim=1)[0]
        cls_attn_maps = (cls_attn - mins[:, None]) / (
            maxs[:, None] - mins[:, None] + 1e-8
        )
        return cls_attn_maps


def find_correspondences(
    image_path1: str,
    image_path2: str,
    num_pairs: int = 10,
    load_size: int = 224,
    layer: int = 9,
    facet: str = "key",
    bin: bool = True,
    thresh: float = 0.05,
    model_type: str = "dino_vits8",
    stride: int = 4,
    return_patches_x_y: bool = True,
) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image
]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :param return_patches_x_y: if True, return patch coordinates.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """

    # extracting descriptors for each image
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # if extractor is None:
    extractor = ViTExtractor(model_type, stride, device=device)

    # image1 descriptors
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(
        image1_batch.to(device), layer, facet, bin
    )
    num_patches1, _ = extractor.num_patches, extractor.load_size

    # image2 descriptors
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(
        image2_batch.to(device), layer, facet, bin
    )
    num_patches2, _ = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(device))[0]

    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=device)
    sim_1, nn_1 = torch.max(
        similarities, dim=-1
    )  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(
        similarities, dim=-2
    )  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(
        num_patches1[0] * num_patches1[1], dtype=torch.bool, device=device
    )
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()

    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)

    n_clusters = min(
        num_pairs, len(all_keys_together)
    )  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together**2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show
    ]  # close bbs
    img1_indices_to_show = torch.arange(
        num_patches1[0] * num_patches1[1], device=device
    )[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]

    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    if return_patches_x_y:
        # make them integers
        img1_y_to_show = (img1_indices_to_show // num_patches1[1]).cpu().numpy()
        img2_y_to_show = (img2_indices_to_show // num_patches2[1]).cpu().numpy()

    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(
        img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show
    ):
        x1_show = (
            (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        )
        y1_show = (
            (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        )
        x2_show = (
            (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        )
        y2_show = (
            (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        )
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))

    if return_patches_x_y:
        return (
            points1,
            points2,
            image1_pil,
            image2_pil,
            [img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show],
            descriptors1,
            descriptors2,
            num_patches1,
        )
    return points1, points2, image1_pil, image2_pil


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y)"""
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def extract_descriptors(
    image_1_path, image_2_path, num_pairs=num_pairs, load_size=load_size
):
    """
    Given a pair of image paths, extracts descriptors and returns the descriptor vectors.
    Inputs: image_1_path, image_2_path: paths to the images.
            num_pairs: number of pairs to extract.
            load_size: size to load the images.
    Outputs: patches_xy: indices of the extracted patches.
              desc1: descriptor map of the first image.
              descriptor_vectors: descriptor vectors of the first image.
              num_patches: number of patches in the x and y direction.
    """
    with torch.no_grad():
        (
            points1,
            points2,
            image1_pil,
            image2_pil,
            patches_xy,
            desc1,
            desc2,
            num_patches,
        ) = find_correspondences(
            image_1_path,
            image_2_path,
            num_pairs,
            load_size,
            layer,
            facet,
            bin,
            thresh,
            model_type,
            stride,
            return_patches_x_y=True,
        )
        # Calculate descriptor dimension from actual tensor shape
        # desc1 has shape [1, 1, num_patches_flat, descriptor_dim]
        descriptor_dim = desc1.shape[-1]
        desc1 = desc1.reshape((num_patches[0], num_patches[1], descriptor_dim))
        descriptor_vectors = desc1[patches_xy[0], patches_xy[1]]

        return patches_xy, desc1, descriptor_vectors, num_patches


def extract_desc_maps(image_paths, load_size=load_size):
    """
    Given a list of image paths, extracts descriptor maps and returns them.
    Inputs: image_paths: list of image paths.
            load_size: size to resize the images.
    Outputs: descriptors_list: list of descriptor maps.
              org_images_list: list of the images.
    """
    if not isinstance(image_paths, list):
        image_paths = [image_paths]
    path = image_paths[0]
    if isinstance(path, str):
        pass
    else:
        paths = []
        for i in range(len(image_paths)):
            paths.append(f"image_{i}.png")
            media.write_image(f"image_{i}.png", image_paths[i])
        image_paths = paths

    extractor = ViTExtractor(model_type, stride, device=device)

    descriptors_list = []
    org_images_list = []
    with torch.no_grad():
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            image_batch, _ = extractor.preprocess(image_path, load_size)
            image_batch_transposed = np.transpose(image_batch[0], (1, 2, 0))

            descriptors = extractor.extract_descriptors(
                image_batch.to(device), layer, facet, bin
            )
            patched_shape = extractor.num_patches
            descriptors = descriptors.reshape((patched_shape[0], patched_shape[1], -1))

            descriptors_list.append(descriptors.cpu())
            image_batch_transposed = (
                image_batch_transposed - image_batch_transposed.min()
            )
            image_batch_transposed = (
                image_batch_transposed / image_batch_transposed.max()
            )
            image_batch_transposed = np.array(
                image_batch_transposed * 255, dtype=np.uint8
            )
            org_images_list.append(
                media.resize_image(
                    image_batch_transposed,
                    (
                        image_batch_transposed.shape[0] // patch_size,
                        image_batch_transposed.shape[1] // patch_size,
                    ),
                )
            )
    return descriptors_list, org_images_list


def extract_descriptor_nn(descriptors, emb_im, patched_shape, return_heatmaps=False):
    """
    Given a list of descriptors and an embedded image, extracts the nearest neighbor descriptor and returns the keypoints.
    Inputs: descriptors: list of descriptor vectors.
            emb_im: embedded image.
            patched_shape: shape of the patches.
            return_heatmaps: if True, returns the heatmaps of the similarities.
    Outputs: cs_ys_list: y coordinates of the keypoints.
             cs_xs_list: x coordinates of the keypoints.
             cs_list: list of heatmaps if return_heatmaps
    """
    cs_ys_list = []
    cs_xs_list = []
    cs_list = []
    cs = torch.nn.CosineSimilarity(dim=-1)
    for i in range(len(descriptors)):
        cs_i = cs(descriptors[i].cpu(), emb_im.cpu())
        cs_i = cs_i.reshape((-1))
        cs_i_y = cs_i.argmax().cpu() // patched_shape[1]
        cs_i_x = cs_i.argmax().cpu() % patched_shape[1]

        cs_ys_list.append(int(cs_i_y) * stride)
        cs_xs_list.append(int(cs_i_x) * stride)

        cs_list.append(np.array(cs_i.cpu()).reshape(patched_shape))

        cs_i = cs_i.reshape(patched_shape)
    if return_heatmaps:
        return cs_ys_list, cs_xs_list, cs_list
    return cs_ys_list, cs_xs_list


def draw_keypoints(image, key_y, key_x, colors):
    """
    Given an image and keypoints, draws the keypoints on the image.
    Inputs: image: image to draw the keypoints on.
            key_y: y coordinates of the keypoints.
            key_x: x coordinates of the keypoints.
            colors: colors of the keypoints.
    Outputs: canvas: image with the keypoints drawn on it
    """
    assert len(key_y) == len(key_x)
    canvas = np.zeros((image.shape[0], image.shape[1], 3))
    if len(image.shape) < 3 or image.shape[-1] == 1:
        canvas[:, :, 0] = image
    else:
        canvas = np.array(image)
    for i in range(len(key_y)):
        color = colors[i]
        canvas = canvas.astype(np.uint8)
        canvas[key_y[i] - 5 : key_y[i] + 5, key_x[i] - 5 : key_x[i] + 5, :] = np.array(
            color
        )
    return canvas


def extract_keypoints_from_masks(
    rgb: np.ndarray,
    masks: np.ndarray,
    intrinsic_matrix: np.ndarray,
    extrinsic_matrix: np.ndarray,
    image_width: int = 848,
    crop_x_offset: int = 184,
    num_candidates_per_mask: int = 5,
    min_dist_bt_keypoints: float = 0.05,
    max_mask_ratio: float = 0.5,
    bounds_min: np.ndarray = None,
    bounds_max: np.ndarray = None,
    device: str = "cuda",
    model_type: str = "dinov3_vits16",
    stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract keypoints from masks using DINOv3 features.

    This function performs the same operation as KeypointProposer.get_keypoints():
    1. Extract DINOv3 features from the image
    2. For each mask, cluster features using k-means
    3. Select keypoint candidates from cluster centers
    4. Convert 2D pixel coordinates to 3D world coordinates using camera parameters
    5. Merge nearby keypoints using MeanShift

    Args:
        rgb: RGB image (H, W, 3) - should be cropped image
        masks: Segmentation masks (H, W) with unique IDs for each object
        intrinsic_matrix: Camera intrinsic matrix (3, 3)
        extrinsic_matrix: Camera extrinsic matrix (4, 4)
        image_width: Width of the original full image before cropping (default: 848)
        crop_x_offset: X offset of the crop in the original image (default: 184)
        num_candidates_per_mask: Number of keypoint candidates per mask
        min_dist_bt_keypoints: Minimum distance between keypoints for merging
        max_mask_ratio: Maximum ratio of mask area to image area
        bounds_min: Minimum workspace bounds [x, y, z]
        bounds_max: Maximum workspace bounds [x, y, z]
        device: Device to run inference on
        model_type: DINOv3 model type
        stride: Stride for feature extraction

    Returns:
        Tuple of (keypoints, projected_image):
        - keypoints: Array of shape (N, 3) with xyz world coordinates
        - projected_image: Visualization image with keypoints drawn
    """
    # Set default bounds if not provided
    if bounds_min is None:
        bounds_min = np.array([0.0, -0.5, 0.0])
    if bounds_max is None:
        bounds_max = np.array([1.0, 0.5, 1.0])

    # Initialize extractor
    extractor = ViTExtractor(model_type=model_type, stride=stride, device=device)

    # Preprocess image
    H, W, _ = rgb.shape

    # Get stride from extractor (it's a tuple for (height, width))
    if isinstance(extractor.stride, tuple):
        stride_h, stride_w = extractor.stride
    else:
        stride_h = stride_w = extractor.stride

    # Calculate number of patches based on stride formula: 1 + (size - patch_size) // stride
    # First align image size to be compatible with patch size
    patch_h = 1 + (H - extractor.p) // stride_h
    patch_w = 1 + (W - extractor.p) // stride_w

    # Calculate the actual image size needed
    new_H = extractor.p + (patch_h - 1) * stride_h
    new_W = extractor.p + (patch_w - 1) * stride_w

    transformed_rgb = cv2.resize(rgb, (new_W, new_H))
    transformed_rgb = transformed_rgb.astype(np.float32) / 255.0

    # Extract features
    rgb_tensor = (
        torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    rgb_tensor = (
        rgb_tensor - torch.tensor(extractor.mean).view(1, 3, 1, 1).to(device)
    ) / torch.tensor(extractor.std).view(1, 3, 1, 1).to(device)

    with torch.no_grad():
        features = extractor.extract_descriptors(
            rgb_tensor, layer=11, facet="key", bin=False
        )

    # Reshape features: [1, 1, num_patches, dim] -> [patch_h, patch_w, dim]
    features_flat = features[0, 0]  # [num_patches, dim]
    features_map = features_flat.reshape(patch_h, patch_w, -1)

    # Resize masks to match feature map size
    masks_resized = cv2.resize(
        masks, (patch_w, patch_h), interpolation=cv2.INTER_NEAREST
    )

    # Extract keypoints for each mask
    candidate_keypoints = []
    candidate_pixels = []
    candidate_rigid_group_ids = []

    unique_masks = np.unique(masks_resized)
    for rigid_group_id, mask_id in enumerate(unique_masks):
        if mask_id == 0:  # Skip background
            continue

        binary_mask = masks_resized == mask_id
        mask_ratio = binary_mask.sum() / (patch_h * patch_w)

        if mask_ratio > max_mask_ratio:
            print(
                f"Skipping mask {mask_id} with ratio {mask_ratio:.3f} > {max_mask_ratio}"
            )
            continue

        # Get features for this mask
        mask_features = features_map[binary_mask]  # [N_mask, dim]

        if len(mask_features) < num_candidates_per_mask:
            n_clusters = len(mask_features)
        else:
            n_clusters = num_candidates_per_mask

        if n_clusters == 0:
            continue

        # Cluster features using k-means
        # Convert to float32 first if needed (BFloat16 not supported by numpy)
        if mask_features.dtype == torch.bfloat16:
            mask_features = mask_features.float()
        mask_features_np = mask_features.cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        kmeans.fit(mask_features_np)

        # For each cluster, find the pixel closest to the cluster center
        for cluster_id in range(n_clusters):
            cluster_mask = kmeans.labels_ == cluster_id

            if cluster_mask.sum() == 0:
                continue

            # Find pixel location in feature map
            mask_indices = np.where(binary_mask)
            pixel_y_feature = mask_indices[0][np.where(cluster_mask)[0]]
            pixel_x_feature = mask_indices[1][np.where(cluster_mask)[0]]

            # Find point closest to cluster center in feature space
            cluster_features = mask_features_np[cluster_mask]
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(cluster_features - center, axis=1)
            closest_idx = np.argmin(distances)

            # Get the pixel location in feature map
            pixel_y_feat = pixel_y_feature[closest_idx]
            pixel_x_feat = pixel_x_feature[closest_idx]

            # Scale back to original cropped image size
            pixel_y_orig = int(pixel_y_feat * H / patch_h)
            pixel_x_orig = int(pixel_x_feat * W / patch_w)

            # Store pixel for later conversion
            candidate_pixels.append([pixel_y_orig, pixel_x_orig])
            candidate_rigid_group_ids.append(rigid_group_id)

    if len(candidate_pixels) == 0:
        print("No candidate keypoints found!")
        return np.array([]), rgb

    # Convert 2D pixels to 3D world coordinates
    print(f"Converting {len(candidate_pixels)} 2D keypoints to 3D world coordinates...")
    for pixel_y, pixel_x in candidate_pixels:
        # Convert cropped image coordinates to full image coordinates
        # The cropped image starts at x_offset in the full image
        full_image_y = pixel_y
        full_image_x = pixel_x + crop_x_offset

        # Convert to normalized coordinates [0-1000] relative to full image
        y_1000 = int((full_image_y / H) * 1000)
        x_1000 = int((full_image_x / image_width) * 1000)

        # Convert pixel to world coordinates using camera parameters
        # Note: Using default radius=3 as in the reference code
        world_coord = tru.pixel_to_world_normalized(
            np.array([y_1000, x_1000]),
            intrinsic_matrix,
            extrinsic_matrix,
            width=image_width,
            radius=3,
        )
        candidate_keypoints.append(world_coord)

    candidate_keypoints = np.array(candidate_keypoints)
    candidate_pixels = np.array(candidate_pixels)
    candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

    # Filter by workspace bounds
    within_bounds_mask = (
        (candidate_keypoints[:, 0] >= bounds_min[0])
        & (candidate_keypoints[:, 0] <= bounds_max[0])
        & (candidate_keypoints[:, 1] >= bounds_min[1])
        & (candidate_keypoints[:, 1] <= bounds_max[1])
        & (candidate_keypoints[:, 2] >= bounds_min[2])
        & (candidate_keypoints[:, 2] <= bounds_max[2])
    )

    candidate_keypoints = candidate_keypoints[within_bounds_mask]
    candidate_pixels = candidate_pixels[within_bounds_mask]
    candidate_rigid_group_ids = candidate_rigid_group_ids[within_bounds_mask]

    if len(candidate_keypoints) == 0:
        print("No keypoints remain after workspace filtering!")
        return np.array([]), rgb

    # Merge close keypoints using MeanShift
    if len(candidate_keypoints) > 1:
        mean_shift = MeanShift(bandwidth=min_dist_bt_keypoints, bin_seeding=True)
        mean_shift.fit(candidate_keypoints)
        cluster_centers = mean_shift.cluster_centers_

        # Select one keypoint per cluster (closest to center)
        merged_indices = []
        for center in cluster_centers:
            distances = np.linalg.norm(candidate_keypoints - center, axis=1)
            closest_idx = np.argmin(distances)
            merged_indices.append(closest_idx)

        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]

    # Sort by pixel location
    sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
    candidate_keypoints = candidate_keypoints[sort_idx]
    candidate_pixels = candidate_pixels[sort_idx]

    # Create visualization
    projected = rgb.copy()
    for keypoint_count, pixel in enumerate(candidate_pixels):
        displayed_text = f"{keypoint_count}"
        text_length = len(displayed_text)

        # Draw box
        box_width = 30 + 10 * (text_length - 1)
        box_height = 30
        cv2.rectangle(
            projected,
            (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
            (pixel[1] + box_width // 2, pixel[0] + box_height // 2),
            (255, 255, 255),
            -1,
        )
        cv2.rectangle(
            projected,
            (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
            (pixel[1] + box_width // 2, pixel[0] + box_height // 2),
            (0, 0, 0),
            2,
        )

        # Draw text
        org = (pixel[1] - 7 * text_length, pixel[0] + 7)
        cv2.putText(
            projected,
            displayed_text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return candidate_keypoints, projected
