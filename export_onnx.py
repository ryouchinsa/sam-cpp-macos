import os
import time
import argparse
import cv2
import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn
from typing import Any
from onnxsim import simplify
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

    @torch.no_grad()
    def forward(
        self, 
        input: torch.Tensor
    ):
        backbone_out = self.model.forward_image(input)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        image_embeddings = feats[2]
        high_res_features1 = feats[0]
        high_res_features2 = feats[1]
        return image_embeddings, high_res_features1, high_res_features2

class ImageDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,  # [1,256,64,64]
        high_res_features1: torch.Tensor, # [1, 32, 256, 256]
        high_res_features2: torch.Tensor, # [1, 64, 128, 128]
        point_coords: torch.Tensor, # [num_labels,num_points,2]
        point_labels: torch.Tensor, # [num_labels,num_points]
        mask_input: torch.Tensor,  # [1,1,256,256]
        has_mask_input: torch.Tensor,  # [1]
        orig_im_size: torch.Tensor   # [2]
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)
        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=[high_res_features1, high_res_features2],
        )
        low_res_masks = torch.clamp(masks, -32.0, 32.0)
        masks = torch.nn.functional.interpolate(
            masks,
            size=(orig_im_size[0], orig_im_size[1]),
            mode="bilinear",
            align_corners=False,
        )
        return masks, iou_predictions, low_res_masks

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:

        point_coords = point_coords + 0.5

        padding_point = torch.zeros(
            (point_coords.shape[0], 1, 2), device=point_coords.device
        )
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device
        )
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords
        )
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight
            * (point_labels == -1)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

def export_image_encoder(model, onnx_path):
    onnx_path = onnx_path + [x for x in onnx_path.split('/') if x][-1] + "_preprocess.onnx"
    input_img = torch.randn(1, 3,1024, 1024)
    out = model(input_img)
    output_names = ["image_embeddings","high_res_features1","high_res_features2"]
    torch.onnx.export(
        model,
        input_img,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=output_names,
    )
    original_model = onnx.load(onnx_path)
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

def export_image_decoder(model, onnx_path):
    onnx_path = onnx_path + [x for x in onnx_path.split('/') if x][-1] + ".onnx"
    image_embeddings = torch.randn(1,256,64,64)
    high_res_features1 = torch.randn(1,32,256,256)
    high_res_features2 = torch.randn(1,64,128,128)
    point_coords = torch.randn(1,2,2)
    point_labels = torch.randn(1,2)
    mask_input = torch.randn(1, 1, 256, 256, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    orig_im_size = torch.tensor([1024,1024],dtype=torch.int64)
    out = model(
        image_embeddings = image_embeddings,
        high_res_features1 = high_res_features1,
        high_res_features2 = high_res_features2,
        point_coords = point_coords,
        point_labels = point_labels,
        mask_input = mask_input,
        has_mask_input = has_mask_input,
        orig_im_size = orig_im_size
    )
    input_name = [
        "image_embeddings",
        "high_res_features1",
        "high_res_features2",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
        "orig_im_size"
    ]
    output_name = ["masks", "iou_predictions", "low_res_masks"]
    dynamic_axes = {
        "point_coords":{0: "num_labels",1:"num_points"},
        "point_labels": {0: "num_labels",1:"num_points"},
        "mask_input": {0: "num_labels"},
        "has_mask_input": {0: "num_labels"}
    }
    torch.onnx.export(
        model,
        (
            image_embeddings,
            high_res_features1,
            high_res_features2,
            point_coords,
            point_labels,
            mask_input,
            has_mask_input,
            orig_im_size
        ),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    original_model = onnx.load(onnx_path)
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, onnx_path)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

def import_onnx(args):
    onnx_path = args.outdir
    encoder_path = onnx_path + [x for x in onnx_path.split('/') if x][-1] + "_preprocess.onnx"
    print(onnxruntime.get_available_providers())
    if torch.cuda.is_available():
        providers=["CUDAExecutionProvider"]
    else:
        providers=["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(
        encoder_path, providers=providers
    )
    model_inputs = session.get_inputs()
    input_names = [
        model_inputs[i].name for i in range(len(model_inputs))
    ]
    input_shape = model_inputs[0].shape
    input_size = input_shape[2:]
    model_outputs = session.get_outputs()
    output_names = [
        model_outputs[i].name for i in range(len(model_outputs))
    ]
    print(input_shape)
    print(input_names)
    print(output_names)
    image = cv2.imread(args.image)
    image_size = image.shape[:2]
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_size[1], input_size[0]))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = (input_img / 255.0 - mean) / std
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
    print("encoder start")
    start = time.perf_counter()
    image_embeddings, high_res_features1, high_res_features2 = session.run(
        output_names, {input_names[0]: input_tensor}
    )
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")

    decoder_path = onnx_path + [x for x in onnx_path.split('/') if x][-1] + ".onnx"
    sessionDecoder = onnxruntime.InferenceSession(
        # decoder_path, providers=onnxruntime.get_available_providers()
        decoder_path, providers=["CPUExecutionProvider"]
    )
    model_inputs_decoder = sessionDecoder.get_inputs()
    input_names_decoder = [
        model_inputs_decoder[i].name for i in range(len(model_inputs_decoder))
    ]
    model_outputs_decoder = sessionDecoder.get_outputs()
    output_names_decoder = [
        model_outputs_decoder[i].name for i in range(len(model_outputs_decoder))
    ]
    print(input_names_decoder)
    print(output_names_decoder)
    mask_input = np.zeros(
        (
            1,
            1,
            input_size[0] // 4,
            input_size[1] // 4,
        ),
        dtype=np.float32,
    )
    has_mask_input = np.array([0], dtype=np.float32)

    points0 = []
    labels0 = []
    points0.append([1215, 125])
    points0.append([1723, 561])
    labels0.append(2)
    labels0.append(3)
    points1 = []
    labels1 = []
    points1.append([890, 85])
    points1.append([1205, 545])
    labels1.append(2)
    labels1.append(3)
    points_batch = [points0, points1]
    labels_batch = [labels0, labels1]
    decode_batch("mask_box_batch.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points_batch, labels_batch, image_embeddings, high_res_features1, high_res_features2)

    points = []
    labels = []
    points.append([1215, 125])
    points.append([1723, 561])
    labels.append(2)
    labels.append(3)
    decode("mask_box1.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)

    points = []
    labels = []
    points.append([890, 85])
    points.append([1205, 545])
    labels.append(2)
    labels.append(3)
    decode("mask_box2.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)

    points = []
    labels = []
    points.append([1255, 360])
    labels.append(1)
    points.append([1500, 420])
    labels.append(1)
    decode("mask_point12.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)
    
    points = []
    labels = []
    points.append([1500, 420])
    labels.append(1)
    decode("mask_point2.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)
    
    points = []
    labels = []
    points.append([1255, 360])
    labels.append(1)
    low_res_mask = decode("mask_point1.png", mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)
    
    points.append([1500, 420])
    labels.append(1)
    has_mask_input = np.array([1], dtype=np.float32)
    decode("mask_point1_then_point2.png", low_res_mask, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2)

def decode_batch(mask_path, mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points_batch, labels_batch, image_embeddings, high_res_features1, high_res_features2):
    batch_num = len(points_batch)
    for i in range(batch_num):
        points, labels = np.array(points_batch[i]), np.array(labels_batch[i])
        points, labels = prepare_points(points, labels, image_size, input_size)
        if i == 0:
            input_point_coords = points
            input_point_labels = labels
        else:
            input_point_coords = np.append(input_point_coords, points, axis=0)
            input_point_labels = np.append(input_point_labels, labels, axis=0)
    orig_im_size = np.array(image_size, dtype=np.int64)
    inputs = [
        image_embeddings, 
        high_res_features1, 
        high_res_features2,
        input_point_coords, 
        input_point_labels, 
        mask_input, 
        has_mask_input, 
        orig_im_size
    ]
    print("decoder start")
    start = time.perf_counter()
    outputs = sessionDecoder.run(
        output_names_decoder,
        {
            input_names_decoder[i]: inputs[i]
            for i in range(len(input_names_decoder))
        },
    )
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
    masks = outputs[0]
    scores = outputs[1]
    batch_num = masks.shape[0]
    mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=np.uint8)
    for i in range(batch_num):
        max_idx = np.argmax(scores[i])
        m = masks[i][max_idx]
        mask[m > 0.0] = 255
    cv2.imwrite(mask_path, mask)

def decode(mask_path, mask_input, has_mask_input, input_size, image_size, sessionDecoder, input_names_decoder, output_names_decoder, points, labels, image_embeddings, high_res_features1, high_res_features2):
    points, labels = np.array(points), np.array(labels)
    input_point_coords, input_point_labels = prepare_points(
        points, labels, image_size, input_size
    )
    orig_im_size = np.array(image_size, dtype=np.int64)
    inputs = [
        image_embeddings, 
        high_res_features1, 
        high_res_features2,
        input_point_coords, 
        input_point_labels, 
        mask_input, 
        has_mask_input, 
        orig_im_size
    ]
    print("decoder start")
    start = time.perf_counter()
    outputs = sessionDecoder.run(
        output_names_decoder,
        {
            input_names_decoder[i]: inputs[i]
            for i in range(len(input_names_decoder))
        },
    )
    print(f"infer time: {(time.perf_counter() - start) * 1000:.2f} ms")
    masks = outputs[0]
    scores = outputs[1]
    low_res_masks = outputs[2]
    max_idx = np.argmax(scores[0])
    mask = masks[0][max_idx]
    mask[mask > 0.0] = 255
    cv2.imwrite(mask_path, mask)
    low_res_mask = low_res_masks[0][max_idx]
    low_res_mask = np.array([[low_res_mask]])
    return low_res_mask

def prepare_points(
        point_coords: list[np.ndarray] | np.ndarray,
        point_labels: list[np.ndarray] | np.ndarray,
        image_size, input_size
) -> tuple[np.ndarray, np.ndarray]:
    input_point_coords = point_coords[np.newaxis, ...]
    input_point_labels = point_labels[np.newaxis, ...]
    input_point_coords[..., 0] = (
        input_point_coords[..., 0]
        / image_size[1]
        * input_size[1]
    )
    input_point_coords[..., 1] = (
        input_point_coords[..., 1]
        / image_size[0]
        * input_size[0]
    )
    return input_point_coords.astype(np.float32), input_point_labels.astype(
        np.float32
    )

def export_onnx(args):
    print(args.config)
    print(args.checkpoint)
    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")

    image_encoder = ImageEncoder(sam2_model)
    export_image_encoder(image_encoder, args.outdir)

    image_decoder = ImageDecoder(sam2_model)
    export_image_decoder(image_decoder, args.outdir)

model_idx = 0
model_type = ["tiny","small","base_plus","large"][model_idx]
config_type = ["t","s","b+","l"][model_idx]
version_idx = 1
version_type = ["2","2.1"][version_idx]
onnx_output_path = "checkpoints/sam{}_{}/".format(version_type, model_type)
model_config_file = "sam{}_hiera_{}.yaml".format(version_type, config_type)
model_checkpoints_file = "checkpoints/sam{}_hiera_{}.pt".format(version_type, model_type)
if not os.path.exists(onnx_output_path):
    os.makedirs(onnx_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--outdir",type=str,default=onnx_output_path,required=False,help="path")
    parser.add_argument("--config",type=str,default=model_config_file,required=False,help="*.yaml")
    parser.add_argument("--checkpoint",type=str,default=model_checkpoints_file,required=False,help="*.pt")
    parser.add_argument("--mode",type=str,default="export",required=False,help="export or import")
    parser.add_argument("--image",type=str,default="david-tomaseti-Vw2HZQ1FGjU-unsplash.jpg",required=False,help="image path")
    args = parser.parse_args()
    if args.mode == "export":
        export_onnx(args)
    else:
        import_onnx(args)

