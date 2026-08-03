"""
scripts/verify_hypotheses.py
Monolithic diagnostic script to verify DataLoader bottlenecks, 
triplet mining behavior, and the effect of horizontal flips.
"""

import sys
import time
import copy
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from zebraid.data.dataset import ZebraDataset, POP_A
from zebraid.data.transforms import train_transforms
from zebraid.models.loss import TripletLossWithMining


def test_horizontal_flips(base_cfg):
    print("\n=== [Experiment A] RandomHorizontalFlip Degradation ===")
    from zebraid.models.train import train
    
    # We will run 1 epoch for speed during verification
    cfg_off = copy.deepcopy(base_cfg)
    if "data" not in cfg_off: cfg_off["data"] = {}
    cfg_off["data"]["use_horizontal_flip"] = False
    cfg_off["training"]["num_epochs"] = 1
    cfg_off["paths"]["checkpoint_dir"] = "checkpoints/verify_A_off"
    
    cfg_on = copy.deepcopy(base_cfg)
    if "data" not in cfg_on: cfg_on["data"] = {}
    cfg_on["data"]["use_horizontal_flip"] = True
    cfg_on["training"]["num_epochs"] = 1
    cfg_on["paths"]["checkpoint_dir"] = "checkpoints/verify_A_on"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f_off:
        yaml.dump(cfg_off, f_off)
        off_path = f_off.name
        
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f_on:
        yaml.dump(cfg_on, f_on)
        on_path = f_on.name
        
    print(">> Running 1 epoch with Flips OFF...")
    res_off = train(config_path=off_path, mode="baseline_a", num_epochs=1)
    
    print(">> Running 1 epoch with Flips ON...")
    res_on = train(config_path=on_path, mode="baseline_a", num_epochs=1)
    
    print("\n--- Experiment A Results ---")
    print(f"Flips OFF -> Rank-1 A: {res_off['rank1_a']:.4f}, mAP A: {res_off['map_a']:.4f}")
    print(f"Flips ON  -> Rank-1 A: {res_on['rank1_a']:.4f}, mAP A: {res_on['map_a']:.4f}")
    
    if res_off['rank1_a'] > res_on['rank1_a']:
        print("💡 Hypothesis CONFIRMED: Horizontal flips degrade Re-ID performance.")
    else:
        print("⚠️ Warning: Could not definitively prove flip degradation in 1 epoch.")


def test_dataloader_workers(cfg):
    print("\n=== [Experiment C] DataLoader Worker Bottleneck ===")
    img_size = cfg["model"].get("img_size", 224)
    
    try:
        dataset = ZebraDataset(
            root=cfg["datasets"]["population_a"]["root"],
            annotation_file=cfg["datasets"]["population_a"]["ann_file"],
            population_label=POP_A,
            split="train",
            transform=train_transforms(img_size),
            train_ratio=1.0 
        )
    except FileNotFoundError:
        print("⚠️ Skipping dataloader test: Dataset not found locally.")
        return

    batch_size = cfg["training"].get("batch_size", 8)
    
    for nw in [0, 2, 4]:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, shuffle=True)
        print(f"Testing num_workers={nw}...")
        start = time.time()
        count = 0
        for i, batch in enumerate(loader):
            # simulate forward/backward pass time
            time.sleep(0.01) 
            count += 1
            if count >= 10:
                break
        elapsed = time.time() - start
        if count > 0:
            print(f"  -> {count} batches took {elapsed:.2f}s ({elapsed/count:.4f}s per batch)")


def test_triplet_mining():
    print("\n=== [Experiment D] Triplet Mining Diagnostics ===")
    loss_fn = TripletLossWithMining(margin=0.3, mining_type="hard")
    
    # Simulate a batch: K=2 instances per individual
    embs = torch.randn(8, 512)
    labels = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4])
    
    loss = loss_fn(embs, labels)
    
    # In PyTorch Metric Learning, the miner holds the last mined indices.
    # The number of valid hard triplets found in the batch:
    try:
        num_triplets = loss_fn.miner.num_triplets
    except AttributeError:
        # Fallback if miner object doesn't expose it
        a, p, n = loss_fn.miner(embs, labels)
        num_triplets = len(a)
        
    print(f"Simulated Batch (8 samples, K=2 per ID):")
    print(f"  -> Triplet Loss: {loss.item():.4f}")
    print(f"  -> Active Hard Triplets Mined: {num_triplets}")
    if num_triplets > 0:
        print("💡 Triplet Mining is actively generating valid triplets.")
    else:
        print("⚠️ Warning: Triplet Mining yielded 0 triplets.")


if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    test_triplet_mining()
    test_dataloader_workers(cfg)
    # Note: test_horizontal_flips takes ~2 mins per epoch. 
    # Uncomment locally to verify:
    # test_horizontal_flips(cfg)
    
    print("\n✅ Verification complete.")
