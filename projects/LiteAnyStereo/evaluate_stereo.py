from __future__ import print_function, division

import argparse
import logging
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import core.stereo_datasets as datasets
from core.models import (
    build_model,
    load_model_weights,
    model_label,
    normalize_model_size,
    normalize_version,
    require_checkpoint,
    resolve_checkpoint,
)
from core.utils.utils import InputPadder
from training.data import JMPLF6020Dataset, TraditionStereoEvaluationDataset
from training.engine import validate
from training.metrics import TRADITION_EXCLUDED_SCENES


VALID_DATASETS = ["jmp", "eth3d", "kitti", "drivingstereo"] + [f"middlebury_{s}" for s in "FHQ"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def runtime_string(elapsed_list):
    if not elapsed_list:
        return " runtime n/a"
    avg_runtime = float(np.mean(elapsed_list))
    return f" {format(1 / avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)"


@torch.no_grad()
def validate_jmp(
    model,
    device,
    max_disp,
    data_root,
    tradition_eval_root,
    split,
    workers,
    evaluation_protocol,
    excluded_scenes,
    epe_threshold,
    save_vis_dir,
    vis_error_max,
):
    """Validate JMP with either native full-label or tradition_stereo protocol."""
    if evaluation_protocol == "tradition":
        dataset = TraditionStereoEvaluationDataset(tradition_eval_root, image_root=data_root)
    else:
        dataset = JMPLF6020Dataset(data_root, split=split, max_disp=max_disp, training=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    metrics = validate(
        model,
        loader,
        device=device,
        max_disp=max_disp,
        amp=False,
        logger=logging.getLogger("liteanystereo.evaluate"),
        evaluation_protocol=evaluation_protocol,
        excluded_scenes=excluded_scenes,
        epe_threshold=epe_threshold,
        save_vis_dir=save_vis_dir,
        vis_error_max=vis_error_max,
    )
    print(
        f"Validation JMP-LF6020 {evaluation_protocol}: EPE {metrics['epe']:.6f}, "
        f"D1 {metrics['d1']:.6f}, Bad1 {metrics['bad1']:.6f}, "
        f"Bad2 {metrics['bad2']:.6f}, Bad3 {metrics['bad3']:.6f}, "
        f"scenes {metrics['scene_count']}, valid pixels {metrics['valid_pixels']}"
    )
    if "vis_output_dir" in metrics:
        print(f"Visualization output: {metrics['vis_output_dir']}")
    return metrics


@torch.no_grad()
def validate_eth3d(model, device, max_disp):
    """Perform validation using the ETH3D train split."""
    model.eval()
    val_dataset = datasets.ETH3D({})

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, imageR_file, GT_file), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, max_disp=max_disp, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()

        occ_mask = Image.open(GT_file.replace("disp0GT.pfm", "mask0nocc.png"))
        occ_mask = np.ascontiguousarray(occ_mask).flatten()
        val = (valid_gt.flatten() >= 0.5) & (occ_mask == 255)

        image_out = (epe_flattened > 1.0)[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            "ETH3D %d/%d. EPE %.4f Bad1 %.4f",
            val_id + 1,
            len(val_dataset),
            image_epe,
            image_out,
        )
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe = float(np.mean(epe_list))
    bad1 = 100 * float(np.mean(out_list))
    print("Validation ETH3D: EPE %f, Bad1 %f" % (epe, bad1))
    return {"eth3d-epe": epe, "eth3d-d1": bad1}


@torch.no_grad()
def validate_kitti(model, device, max_disp, year=2015):
    """Perform validation using the KITTI train split."""
    model.eval()
    val_dataset = datasets.KITTI({}, image_set="training", year=year)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    d1_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        imageL_file, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        flow_pr = model(image1, image2, max_disp=max_disp, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if val_id > 50:
            elapsed_list.append(elapsed)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        flow_gt_flat = flow_gt.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < max_disp)

        rel_error = epe_flattened / torch.clamp(flow_gt_flat.abs(), min=1e-5)
        d1_mask = (epe_flattened > 3.0) & (rel_error > 0.05)

        epe_list.append(epe_flattened[val].mean().item())
        d1_list.append(d1_mask[val].cpu().numpy())

    epe = float(np.mean(epe_list))
    d1 = 100 * float(np.mean(np.concatenate(d1_list)))
    print(f"Validation KITTI {year}: EPE {epe}, D1 {d1}" + runtime_string(elapsed_list))
    return {"kitti-epe": epe, "kitti-d1": d1}


@torch.no_grad()
def validate_middlebury(model, device, max_disp, split="MiddEval3", resolution="F"):
    """Perform validation using the Middlebury-v3 dataset."""
    model.eval()
    val_dataset = datasets.Middlebury({}, split=split, resolution=resolution)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        flow_pr = model(image1, image2, max_disp=max_disp, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        occ_mask = Image.open(imageL_file.replace("im0.png", "mask0nocc.png")).convert("L")
        occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()

        val = (valid_gt.reshape(-1) >= 0.5) & (flow_gt[0].reshape(-1) < max_disp) & (occ_mask == 255)
        image_out = (epe_flattened > 2.0)[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(
            "Middlebury %d/%d. EPE %.4f Bad2 %.4f",
            val_id + 1,
            len(val_dataset),
            image_epe,
            image_out,
        )
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe = float(np.mean(epe_list))
    bad2 = 100 * float(np.mean(out_list))
    print(f"Validation Middlebury{split}_{resolution}_{max_disp}: EPE {epe}, Bad2 {bad2}")
    return {f"middlebury{split}_{resolution}-epe": epe, f"middlebury{split}-d1": bad2}


@torch.no_grad()
def validate_driving(model, device, max_disp, split="cloudy"):
    """Perform validation using the DrivingStereo weather split."""
    model.eval()
    val_dataset = datasets.DrivingStereoWeather({}, image_set=split)
    torch.backends.cudnn.benchmark = device.type == "cuda"

    d1_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].to(device)
        image2 = image2[None].to(device)

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        flow_pr = model(image1, image2, max_disp=max_disp, test_mode=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if val_id > 50:
            elapsed_list.append(elapsed)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)

        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()
        epe_flattened = epe.flatten()
        flow_gt_flat = flow_gt.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < max_disp)

        rel_error = epe_flattened / torch.clamp(flow_gt_flat.abs(), min=1e-5)
        d1_mask = (epe_flattened > 3.0) & (rel_error > 0.05)

        epe_list.append(epe_flattened[val].mean().item())
        d1_list.append(d1_mask[val].cpu().numpy())

    epe = float(np.mean(epe_list))
    d1 = 100 * float(np.mean(np.concatenate(d1_list)))
    print(f"Validation DrivingStereo {split}: EPE {epe}, D1 {d1}" + runtime_string(elapsed_list))
    return {"drivingstereo-epe": epe, "drivingstereo-d1": d1}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LiteAnyStereo LAS1 or LAS2.")
    parser.add_argument("--version", default="las1", help="model version: las1/v1 or las2/v2")
    parser.add_argument("--model_size", "--model-size", default=None, help="LAS2 model size: s, m, l, or h")
    parser.add_argument("--restore_ckpt", default=None, help="checkpoint path; use 'none' to skip loading")
    parser.add_argument("--dataset", default="middlebury_H", choices=VALID_DATASETS)
    parser.add_argument("--data_root", default=None, help="JMP-LF6020 root for --dataset jmp")
    parser.add_argument("--split", choices=["train", "val"], default="val", help="JMP split to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="JMP data-loading workers")
    parser.add_argument(
        "--evaluation_protocol",
        choices=["auto", "standard", "tradition"],
        default="auto",
    )
    parser.add_argument(
        "--tradition_eval_root",
        default="../tradition_stereo/datasets/FDJYP-3",
    )
    parser.add_argument("--epe_threshold", type=float, default=20.0)
    parser.add_argument("--no_epe_filter", action="store_true")
    parser.add_argument("--include_excluded", action="store_true")
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="save prediction vis.png and diagnostic comparison.png files per JMP scene",
    )
    parser.add_argument(
        "--vis_dir",
        default="./runs/evaluation/jmp_evaluation",
        help="root directory used by --save_vis",
    )
    parser.add_argument("--vis_error_max", type=float, default=20.0)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--max_disp", type=int, default=192, help="maximum disparity for evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.evaluation_protocol == "auto":
        args.evaluation_protocol = "tradition" if args.dataset == "jmp" else "standard"
    if args.dataset == "jmp" and args.evaluation_protocol == "standard" and not args.data_root:
        raise ValueError("--data_root is required for standard JMP evaluation")
    if args.dataset == "jmp" and args.evaluation_protocol == "tradition":
        if not args.tradition_eval_root or not args.data_root:
            raise ValueError(
                "--tradition_eval_root and rectified --data_root are required for tradition JMP evaluation"
            )
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.epe_threshold <= 0:
        raise ValueError("--epe_threshold must be positive")
    if args.vis_error_max <= 0:
        raise ValueError("--vis_error_max must be positive")
    if args.save_vis and args.dataset != "jmp":
        raise ValueError("--save_vis is currently supported for dataset=jmp")
    version = normalize_version(args.version)
    model_size = normalize_model_size(version, args.model_size)
    label = model_label(version, model_size)
    ckpt_path = resolve_checkpoint(version, args.restore_ckpt, model_size=model_size)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    model = build_model(version, fnet_pretrained=False, model_size=model_size, max_disp=args.max_disp)

    if ckpt_path is not None:
        require_checkpoint(ckpt_path)
        logging.info("Loading %s checkpoint from %s", label, ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        load_model_weights(model, checkpoint, strict=True)
        logging.info("Done loading checkpoint")

    model.to(device)
    model.eval()

    print(f"Model: {label}")
    if hasattr(model, "fnet_name"):
        print(f"FeatureNet: {model.fnet_name} {getattr(model, 'fnet_channels', '')}")
    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")

    if args.dataset == "jmp":
        validate_jmp(
            model,
            device,
            args.max_disp,
            args.data_root,
            args.tradition_eval_root,
            args.split,
            args.workers,
            args.evaluation_protocol,
            () if args.include_excluded else TRADITION_EXCLUDED_SCENES,
            None if args.no_epe_filter else args.epe_threshold,
            args.vis_dir if args.save_vis else None,
            args.vis_error_max,
        )
    elif args.dataset == "eth3d":
        validate_eth3d(model, device, args.max_disp)
    elif args.dataset == "kitti":
        validate_kitti(model, device, args.max_disp, year=2012)
        validate_kitti(model, device, args.max_disp, year=2015)
    elif args.dataset.startswith("middlebury_"):
        validate_middlebury(model, device, args.max_disp, resolution=args.dataset[-1])
    elif args.dataset == "drivingstereo":
        for split in ["cloudy", "foggy", "rainy", "sunny"]:
            validate_driving(model, device, args.max_disp, split=split)


if __name__ == "__main__":
    main()
