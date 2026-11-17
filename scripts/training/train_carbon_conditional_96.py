"""Train PorousDiff continuous load/stress-conditioned 3D diffusion models (single or multi-electrode).

Supports:
  - Single-material binary training (`--states ...`) for backward-compatible Virgin Felt runs,
  - Unified multi-electrode 3-phase training (`--manifest-json bundle_manifest.json`) across
    `virgin_carbon`, `tio2_anode`, and `cuhcf_cathode` with balanced material/state sampling,
  - Few-Shot "Beamtime Multiplier" filtering (`--include-states` / `--exclude-states`) holding
    out intermediate coated-electrode loads while leveraging dense Virgin Felt scaffold physics,
  - Zero-Terminal-SNR `v_prediction`, EMA, GH200 `bf16` AMP, and full RNG `--resume`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from porousdiff.data import (
    MATERIAL_TO_ID,
    LoadConditionedPatchDataset,
    MultiElectrodePatchDataset,
    capture_rng_states,
    default_project_root,
    restore_rng_states,
)
from porousdiff.losses import PhysicsInformedDiffusionLoss
from porousdiff.models import ContinuousLoadConditionedUNet, EMAModel, build_noise_scheduler
from porousdiff.preprocessing import parse_load_value

DEFAULT_CARBON_STATES = {
    "Op008mVV": 0.008,
    "Op018mVV": 0.018,
    "Op033mVV": 0.033,
}


def parse_states_arg(
    entries: list[str] | None, states_json: Path | None
) -> dict[str, float]:
    if states_json is not None:
        with states_json.open("r") as f:
            raw = json.load(f)
        return {str(k): float(v) for k, v in raw.items()}
    if not entries:
        return dict(DEFAULT_CARBON_STATES)

    mapping: dict[str, float] = {}
    for item in entries:
        if "=" in item:
            name, val_s = item.split("=", 1)
            mapping[name.strip()] = float(val_s.strip())
        else:
            name = item.strip()
            mapping[name] = parse_load_value(name)
    return mapping


def load_manifest_specs(
    manifest_path: Path,
    data_root: Path,
    include_states: set[str] | None = None,
    exclude_states: set[str] | None = None,
    include_repeats: bool = False,
    use_stress_kpa: bool = False,
) -> tuple[list[dict[str, object]], int, float]:
    with manifest_path.open("r") as f:
        manifest = json.load(f)

    specs: list[dict[str, object]] = []
    max_phases = 2
    vox_list: list[float] = []

    for scan in manifest.get("scans", []):
        st = str(scan["state"])
        if not include_repeats and bool(scan.get("is_repeat_or_secondary", False)):
            if include_states is None or st not in include_states:
                continue
        if include_states is not None and st not in include_states:
            continue
        if exclude_states is not None and st in exclude_states:
            continue

        if use_stress_kpa and scan.get("stress_kpa") is not None:
            load_val = float(scan["stress_kpa"])
        else:
            load_val = float(scan["load_mv_per_v"])

        eff_vox = float(scan.get("effective_voxel_size_um", manifest.get("effective_voxel_size_um", 1.0)))
        n_ph = int(scan.get("num_phases", 2))
        max_phases = max(max_phases, n_ph)
        vox_list.append(eff_vox)

        patch_dir_name = scan.get("patch_dir_name")
        patch_dir = (data_root / str(patch_dir_name)) if patch_dir_name else None

        specs.append(
            {
                "state": st,
                "material": str(scan.get("material", "virgin_carbon")),
                "load_value": load_val,
                "voxel_size_um": eff_vox,
                "num_phases": n_ph,
                "patch_dir": str(patch_dir) if (patch_dir and patch_dir.is_dir()) else None,
            }
        )

    if len(specs) < 2:
        raise ValueError(f"Manifest filtering yielded < 2 training states ({len(specs)} found)")
    ref_vox = float(np.median(vox_list)) if vox_list else 1.0
    return specs, max_phases, ref_vox


def atomic_torch_save(payload: dict[str, object], target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def main() -> None:
    proj_root = default_project_root()
    parser = argparse.ArgumentParser(
        description="Train continuous load/stress-conditioned 3D diffusion model (PorousDiff)"
    )
    parser.add_argument("--experiment", default="carbon_conditional_96_v2")
    parser.add_argument("--data-root", type=Path, default=proj_root / "data" / "carbon_virgin")
    parser.add_argument("--output-root", type=Path, default=proj_root)
    parser.add_argument("--version", default="carbon_calibrated_v1")
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="Dataset=load pairs (e.g. Op008mVV=0.008 Op018mVV=0.018 Op033mVV=0.033) or folder names",
    )
    parser.add_argument("--states-json", type=Path, default=None)
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Multi-electrode bundle_manifest.json activating material + 3-phase + voxel-size conditioning",
    )
    parser.add_argument("--include-states", nargs="*", default=None, help="Explicit subset of manifest states to train on")
    parser.add_argument("--exclude-states", nargs="*", default=None, help="Held-out manifest states to exclude from training")
    parser.add_argument("--include-repeats", action="store_true", help="Include replicate/secondary scans from manifest")
    parser.add_argument("--use-stress-kpa", action="store_true", help="Condition on calibrated stress_kpa from manifest")
    parser.add_argument("--num-phases", type=int, default=0, choices=[0, 2, 3], help="0=infer from manifest (or 2 for legacy --states)")
    parser.add_argument("--balanced-sampling", action="store_true", help="Equalize gradient weight across materials and states")

    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--channels", default="32,64,128")
    parser.add_argument("--attention-levels", default="0,0,1")
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--norm-groups", type=int, default=16)

    parser.add_argument("--schedule", default="zero_snr_scaled_linear")
    parser.add_argument("--prediction-type", default="v_prediction", choices=["v_prediction", "epsilon"])
    parser.add_argument(
        "--conditioning-mode",
        default="multiscale_fourier",
        choices=["multiscale_fourier", "legacy_crossattn"],
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cfg-drop-prob", type=float, default=0.10)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument(
        "--physics-loss",
        action="store_true",
        help="Enable physics-informed x0_hat training loss (Min-SNR-gamma + 3D interfacial gradient + solid support + catalyst loading conservation)",
    )
    parser.add_argument("--snr-gamma", type=float, default=5.0)
    parser.add_argument("--interface-weight", type=float, default=0.05)
    parser.add_argument("--support-weight", type=float, default=0.05)
    parser.add_argument("--loading-weight", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")

    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0, help="Cap train steps/epoch for smoke tests")
    parser.add_argument("--max-val-steps", type=int, default=0)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment):
        parser.error("Invalid --experiment name")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        parser.error("--epochs, --batch-size, and --grad-accum must be >= 1")
    if not 0.0 <= args.cfg_drop_prob < 1.0:
        parser.error("--cfg-drop-prob must be in [0, 1)")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA GPU required (pass --allow-cpu only for local smoke tests).")

    use_amp = False
    amp_dtype = torch.float32
    if device.type == "cuda" and args.amp != "off":
        if args.amp == "bf16" or (args.amp == "auto" and torch.cuda.is_bf16_supported()):
            use_amp = True
            amp_dtype = torch.bfloat16
        else:
            use_amp = True
            amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    channels = tuple(int(c.strip()) for c in args.channels.split(","))
    attention_levels = tuple(bool(int(a.strip())) for a in args.attention_levels.split(","))
    if len(channels) != len(attention_levels):
        raise ValueError("--channels and --attention-levels must have matching length")

    result_dir = args.output_root / "results" / args.experiment
    checkpoint_dir = args.output_root / "checkpoints" / args.experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'} | "
        f"AMP: {amp_dtype if use_amp else 'fp32'}",
        flush=True,
    )

    pin = device.type == "cuda"
    workers = args.num_workers if device.type == "cuda" else 0
    is_multielectrode = args.manifest_json is not None

    if is_multielectrode:
        inc_set = set(args.include_states) if args.include_states else None
        exc_set = set(args.exclude_states) if args.exclude_states else None
        scan_specs, inferred_phases, ref_vox = load_manifest_specs(
            manifest_path=args.manifest_json,
            data_root=args.data_root,
            include_states=inc_set,
            exclude_states=exc_set,
            include_repeats=args.include_repeats,
            use_stress_kpa=args.use_stress_kpa,
        )
        num_phases = args.num_phases if args.num_phases in (2, 3) else inferred_phases
        num_materials = len(MATERIAL_TO_ID)
        states = {str(s["state"]): float(s["load_value"]) for s in scan_specs}  # type: ignore[arg-type]
        load_min = min(states.values())
        load_max = max(states.values())
        if load_max <= load_min:
            load_max = load_min + 1.0

        train_data_mm = MultiElectrodePatchDataset(
            data_root=args.data_root,
            scan_specs=scan_specs,
            split="train",
            patch_size=args.patch_size,
            version=args.version,
            num_phases=num_phases,
            augment=not args.no_augment,
        )
        val_data_mm = MultiElectrodePatchDataset(
            data_root=args.data_root,
            scan_specs=scan_specs,
            split="val",
            patch_size=args.patch_size,
            version=args.version,
            num_phases=num_phases,
            augment=False,
        )
        print(
            f"Multi-electrode train: {len(train_data_mm)} patches | materials={train_data_mm.material_counts}",
            flush=True,
        )
        print(f"Multi-electrode val:   {len(val_data_mm)} patches", flush=True)

        if args.balanced_sampling:
            weights = train_data_mm.build_balanced_weights()
            sampler = WeightedRandomSampler(weights, num_samples=len(train_data_mm), replacement=True)
            train_loader: DataLoader = DataLoader(
                train_data_mm,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=workers,
                pin_memory=pin,
            )
        else:
            train_loader = DataLoader(
                train_data_mm,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin,
            )
        val_loader: DataLoader = DataLoader(
            val_data_mm,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin,
        )
    else:
        states = parse_states_arg(args.states, args.states_json)
        if len(states) < 2:
            raise ValueError("Conditional training requires at least 2 distinct load states")
        load_min = min(states.values())
        load_max = max(states.values())
        num_phases = args.num_phases if args.num_phases in (2, 3) else 2
        num_materials = 1
        ref_vox = 1.0
        scan_specs = []

        train_data = LoadConditionedPatchDataset(
            data_root=args.data_root,
            states=states,
            split="train",
            patch_size=args.patch_size,
            version=args.version,
            augment=not args.no_augment,
            num_phases=num_phases,
        )
        val_data = LoadConditionedPatchDataset(
            data_root=args.data_root,
            states=states,
            split="val",
            patch_size=args.patch_size,
            version=args.version,
            augment=False,
            num_phases=num_phases,
        )
        print(f"Train patches: {len(train_data)} ({train_data.state_counts})", flush=True)
        print(f"Val patches:   {len(val_data)} ({val_data.state_counts})", flush=True)

        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=pin,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=pin,
        )

    model = ContinuousLoadConditionedUNet(
        channels=channels,
        attention_levels=attention_levels,
        num_res_blocks=args.num_res_blocks,
        norm_num_groups=args.norm_groups,
        load_min=load_min,
        load_max=load_max,
        use_flash_attention=(device.type == "cuda" and use_amp),
        conditioning_mode=args.conditioning_mode,
        num_phases=num_phases,
        num_materials=num_materials,
        ref_voxel_size_um=ref_vox,
    ).to(device)

    ema = EMAModel(model, decay=args.ema_decay)
    scheduler = build_noise_scheduler(
        sampler="ddpm",
        num_train_timesteps=args.timesteps,
        schedule=args.schedule,
        prediction_type=args.prediction_type,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    physics_criterion: PhysicsInformedDiffusionLoss | None = None
    if args.physics_loss:
        physics_criterion = PhysicsInformedDiffusionLoss(
            alphas_cumprod=scheduler.alphas_cumprod,
            prediction_type=args.prediction_type,
            snr_gamma=args.snr_gamma,
            interface_weight=args.interface_weight,
            support_weight=args.support_weight,
            loading_weight=args.loading_weight,
        ).to(device)
        print(
            f"Physics-informed loss enabled: snr_gamma={args.snr_gamma}, "
            f"iface={args.interface_weight}, support={args.support_weight}, loading={args.loading_weight}",
            flush=True,
        )

    model_config = {
        "patch_size": args.patch_size,
        "channels": list(channels),
        "attention_levels": list(attention_levels),
        "num_res_blocks": args.num_res_blocks,
        "norm_groups": args.norm_groups,
        "load_min": load_min,
        "load_max": load_max,
        "schedule": args.schedule,
        "prediction_type": args.prediction_type,
        "conditioning_mode": args.conditioning_mode,
        "num_phases": num_phases,
        "num_materials": num_materials,
        "ref_voxel_size_um": ref_vox,
        "use_stress_kpa": args.use_stress_kpa,
        "physics_loss": args.physics_loss,
        "snr_gamma": args.snr_gamma if args.physics_loss else 0.0,
        "timesteps": args.timesteps,
        "states": states,
        "scan_specs": scan_specs,
        "version": args.version,
    }

    latest_path = checkpoint_dir / "latest.pth"
    best_path = checkpoint_dir / "best_val.pth"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot --resume; missing {latest_path}")
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        optimizer.load_state_dict(saved["optimizer"])
        if "scaler" in saved and scaler.is_enabled():
            scaler.load_state_dict(saved["scaler"])
        history = list(saved["history"])
        start_epoch = int(saved["epoch"]) + 1
        best_val_loss = float(saved.get("best_val_loss", float("inf")))
        restore_rng_states(saved["rng_states"])
        print(f"Resumed from epoch {start_epoch} (best_val={best_val_loss:.6f})", flush=True)
    elif latest_path.exists():
        raise FileExistsError(
            f"{latest_path} already exists; pass --resume or a new --experiment name."
        )

    stop_requested = False

    def _handle_preempt(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        print(f"Received signal {signum}; will checkpoint cleanly at end of epoch.", flush=True)
        stop_requested = True

    signal.signal(signal.SIGTERM, _handle_preempt)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _handle_preempt)

    def _unpack_batch(
        batch: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if len(batch) == 4:
            imgs, lds, mats, voxs = batch
            return (
                imgs.to(device, non_blocking=True),
                lds.to(device, non_blocking=True),
                mats.to(device, non_blocking=True),
                voxs.to(device, non_blocking=True),
            )
        imgs, lds = batch
        return imgs.to(device, non_blocking=True), lds.to(device, non_blocking=True), None, None

    def compute_batch_loss(
        batch: tuple[torch.Tensor, ...],
        is_train: bool,
    ) -> torch.Tensor:
        images, loads, mats, voxs = _unpack_batch(batch)
        bsz = images.shape[0]
        noise = torch.randn_like(images)
        timesteps = torch.randint(0, args.timesteps, (bsz,), device=device).long()
        if is_train and args.cfg_drop_prob > 0.0:
            keep = torch.rand(bsz, device=device) >= args.cfg_drop_prob
            cond_mask = keep.to(torch.float32)
        else:
            cond_mask = torch.ones(bsz, device=device, dtype=torch.float32)

        noisy = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
        target = (
            scheduler.get_velocity(images, noise, timesteps)
            if args.prediction_type == "v_prediction"
            else noise
        )

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(
                noisy,
                timesteps,
                loads,
                cond_mask=cond_mask,
                material_ids=mats,
                voxel_sizes=voxs,
            )
            if physics_criterion is not None:
                loss, _ = physics_criterion(
                    model_pred=pred,
                    target=target,
                    x_t=noisy,
                    x_0=images,
                    timesteps=timesteps,
                )
            else:
                loss = F.mse_loss(pred.float(), target.float())
        return loss

    for epoch in range(start_epoch, args.epochs):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_total = 0.0
        train_steps = 0

        for step_idx, batch in enumerate(train_loader, start=1):
            raw_loss = compute_batch_loss(batch, is_train=True)
            scaled_loss = raw_loss / args.grad_accum

            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            train_total += float(raw_loss.item())
            train_steps += 1

            is_last_batch = (step_idx == len(train_loader)) or (
                bool(args.max_steps) and train_steps >= args.max_steps
            )
            if (train_steps % args.grad_accum == 0) or is_last_batch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

            if train_steps % 50 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} step {train_steps}: loss={raw_loss.item():.6f}",
                    flush=True,
                )
            if args.max_steps and train_steps >= args.max_steps:
                break

        ema_model = ema.shadow_model
        ema_model.eval()
        val_gen = torch.Generator(device=device).manual_seed(args.seed + 10_007)
        val_total = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                images, loads, mats, voxs = _unpack_batch(batch)
                bsz = images.shape[0]
                noise = torch.randn(images.shape, generator=val_gen, device=device, dtype=images.dtype)
                timesteps = torch.randint(
                    0, args.timesteps, (bsz,), generator=val_gen, device=device
                ).long()
                noisy = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                target = (
                    scheduler.get_velocity(images, noise, timesteps)
                    if args.prediction_type == "v_prediction"
                    else noise
                )
                ones = torch.ones(bsz, device=device, dtype=torch.float32)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    pred = ema_model(
                        noisy,
                        timesteps,
                        loads,
                        cond_mask=ones,
                        material_ids=mats,
                        voxel_sizes=voxs,
                    )
                    v_loss = F.mse_loss(pred.float(), target.float())
                val_total += float(v_loss.item())
                val_steps += 1
                if args.max_val_steps and val_steps >= args.max_val_steps:
                    break

        if device.type == "cuda":
            torch.cuda.synchronize()

        val_loss = val_total / max(1, val_steps)
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        record: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": round(train_total / max(1, train_steps), 6),
            "val_loss_ema": round(val_loss, 6),
            "best_val_loss": round(best_val_loss, 6),
            "epoch_seconds": round(time.perf_counter() - started, 2),
            "peak_allocated_gib": (
                round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                if device.type == "cuda"
                else 0.0
            ),
            "peak_reserved_gib": (
                round(torch.cuda.max_memory_reserved() / 1024**3, 3)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

        with (result_dir / "losses.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writeheader()
            writer.writerows(history)

        ckpt_payload: dict[str, object] = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": model_config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "rng_states": capture_rng_states(),
        }

        atomic_torch_save(ckpt_payload, latest_path)
        if is_best:
            atomic_torch_save(ckpt_payload, best_path)
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            atomic_torch_save(ckpt_payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")

        if stop_requested:
            print(f"Clean exit after saving epoch {epoch + 1} checkpoint.", flush=True)
            return

    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()
