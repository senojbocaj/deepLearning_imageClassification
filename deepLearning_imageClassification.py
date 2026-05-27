# MLPs vs CNNs on MNIST & CIFAR-10
# This script runs the full suite of experiments for Project 3, including:
# - Data loading and preprocessing for MNIST and CIFAR-10
# - Model definitions for MLPs and CNNs
# - Training loops with early stopping
# - Hyperparameter search over predefined grids
# - Auto-saving results to survive Colab disconnects
# - Final results tables summarizing test accuracies and runtimes

import json
import time
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

# Reproducibility & Device Setup
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
MNIST_CACHE = None
CIFAR_CACHE = None

# Set PyTorch logging to only show errors to reduce clutter during training. This is especially helpful in Colab where verbose logs can make it hard to track progress.
torch._logging.set_logs(inductor=logging.ERROR)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

# Detect GPU and set up device with optimal settings for performance. Falls back to CPU if no GPU is found.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PrecachedDataset is a custom Dataset wrapper that loads all data into GPU memory at once. 
# This can significantly speed up training by eliminating data transfer overhead during each batch, e
# specially for small datasets like MNIST and CIFAR-10. However, it requires enough GPU memory to hold the
#  entire dataset, so it's best used on machines with ample VRAM.
class PrecachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        # Use a temporary loader to process all images once
        loader = DataLoader(dataset, batch_size=1024, num_workers=0)
        all_x, all_y = [], []

        print(f"  [CACHE] Preloading {len(dataset)} samples to {device}...")
        for x, y in loader:
            all_x.append(x.to(device))
            all_y.append(y.to(device))
            
        self.x = torch.cat(all_x, dim=0)
        self.y = torch.cat(all_y, dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Enable cuDNN autotuner for optimal convolution algorithms on GPU. This can significantly speed up training, 
# especially for CNNs. On CPU, we disable it for deterministic behavior.
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False
    print(f"✓ GPU detected : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM          : "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
else:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print("ERROR: No GPU found — running on CPU. Runtime will be slow.")
    print("  In Colab: Runtime → Change runtime type → T4 GPU")

PIN_MEMORY = DEVICE.type == "cuda"

# Benchmarking utility to time a single training epoch. Useful for sanity-checking GPU performance before running the full hyperparameter search.
def benchmark_one_epoch(model, loader, criterion, optimizer):
    # Warm-up run (compilation, caching, etc.)
    t0 = time.time()
    train_epoch(model, loader, criterion, optimizer)
    print(f"  [benchmark] 1 epoch = {time.time() - t0:.2f}s")

# Data Loading & Preprocessing
def _make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,    

    )

# For MNIST and CIFAR-10, we apply standard normalization and split the training set into train/val subsets. 
# The test set remains untouched for final evaluation.

# MNIST: 60,000 train → 50,000 train / 10,000 val = 10,000 test.
# Normalization: mean=0.1307, std=0.3081 (MNIST dataset standard)
def get_mnist_loaders(batch_size, val_size=10_000):
    global MNIST_CACHE
    
    if MNIST_CACHE is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        raw_train = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
        raw_test = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
        
        # Move everything to DEVICE (GPU) immediately
        MNIST_CACHE = {
            "train": PrecachedDataset(raw_train, DEVICE),
            "test": PrecachedDataset(raw_test, DEVICE)
        }

    train_set, val_set = random_split(
        MNIST_CACHE["train"], [len(MNIST_CACHE["train"]) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    
    # We use num_workers=0 because the data is already on the GPU/RAM
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0),
        DataLoader(MNIST_CACHE["test"], batch_size=512, shuffle=False, num_workers=0),
    )

# CIFAR-10 normalization uses per-channel mean/std computed from the training set. 
# This helps stabilize training and improve convergence for CNNs.

# CIFAR-10: 50,000 train → 45,000 train / 5,000 val | 10,000 test.
# Per-channel normalization (standard CIFAR-10 stats).
def get_cifar10_loaders(batch_size, val_size=5_000):
    global CIFAR_CACHE
    
    if CIFAR_CACHE is None:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        
        raw_train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
        raw_test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
        
        CIFAR_CACHE = {
            "train": PrecachedDataset(raw_train, DEVICE),
            "test": PrecachedDataset(raw_test, DEVICE)
        }

    train_set, val_set = random_split(
        CIFAR_CACHE["train"], [len(CIFAR_CACHE["train"]) - val_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0),
        DataLoader(CIFAR_CACHE["test"], batch_size=512, shuffle=False, num_workers=0),
    )

# Model Definitions
class MLP(nn.Module):
    # Generic MLP: flattens input → (Linear + ReLU + Dropout) × N → logits.
    # No softmax — CrossEntropyLoss applies it internally.

    def __init__(self, input_dim, hidden_sizes, num_classes, dropout=0.0):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

# Factory function to build MLPs of varying depth based on the specified architecture. 
# This allows us to easily create shallow, medium, and deep MLPs with different hidden layer configurations.
def build_mlp(depth, input_dim, num_classes, dropout):
    arch = {
        "shallow": [128],
        "medium":  [512, 256, 128],
        "deep":    [1024, 512, 256, 128, 64],
    }
    return MLP(input_dim, arch[depth], num_classes, dropout)

# Two CNN architectures: a simple one with 2 conv layers and an enhanced one with 3 conv layers and more filters.
class SimpleCNN(nn.Module):
    # 2 conv layers (8 filters each), BN + MaxPool, FC head.
    # Supports 1-channel (MNIST) and 3-channel (CIFAR-10).

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),  nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 8,     kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),  nn.ReLU(), nn.MaxPool2d(2),
        )
        self.num_classes = num_classes
        self.classifier  = None

    # Dynamically builds the fully connected head based on the output dimension of the conv layers. 
    # This allows us to avoid hardcoding the input size for the classifier, making it adaptable to different input image sizes and architectures.
    def _build_head(self, x):
        with torch.no_grad():
            dim = self.features(x[:1]).flatten(1).shape[1]
        self.classifier = nn.Linear(dim, self.num_classes).to(x.device)

    # During the first forward pass, we check if the classifier head is built. 
    # If not, we run a dummy input through the conv layers to determine the output dimension 
    # and build the head accordingly. This lazy initialization allows us to keep the model definition clean and flexible.
    def forward(self, x):
        if self.classifier is None:
            self._build_head(x)
        return self.classifier(self.features(x).flatten(1))

# The enhanced CNN adds a third conv layer with more filters (64) and preserves spatial resolution by skipping the final MaxPool.
class EnhancedCNN(nn.Module):
    # 3 conv layers with increasing filters (16→32→64), BN + MaxPool, FC head.
    # Uses 3×3 kernels and stride=1 throughout (per project spec).

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32,          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,          kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), 
        )
        self.num_classes = num_classes
        self.classifier  = None

    def _build_head(self, x):
        with torch.no_grad():
            dim = self.features(x[:1]).flatten(1).shape[1]
        self.classifier = nn.Linear(dim, self.num_classes).to(x.device)

    def forward(self, x):
        if self.classifier is None:
            self._build_head(x)
        return self.classifier(self.features(x).flatten(1))

# Model Compilation (PyTorch 2.0+)
def compile_model(model):
    if DEVICE.type == "cuda" and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception as e:
            print(f"  [compile] skipped ({e})")
    return model

# Training & Evaluation Loops
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    total_loss, correct, total = 0.0, 0, 0

    for X, y in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True) 

        with torch.amp.autocast('cuda'):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

# Evaluation loop runs in no_grad mode for efficiency, and computes overall accuracy across the dataset.
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        correct += (model(X).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

# Training loop with early stopping based on validation accuracy. Saves the best model state and returns it along with the best validation accuracy, stopping epoch, and elapsed time.
def run_training(model, train_loader, val_loader, epochs=10, lr=1e-3, optimizer_name="adam", weight_decay=0.0, patience=3):
    # Trains with early stopping (patience on val accuracy).
    # Returns: best_val_acc, best_state_dict, stopping_epoch, elapsed_seconds

    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adam"
        else optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                       weight_decay=weight_decay)
    )

    best_val_acc, best_state, no_improve = 0.0, None, 0
    t0 = time.time()

    # We print training progress for each epoch, including the training loss, training accuracy, and validation accuracy.
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_acc               = evaluate(model, val_loader)
        print(f"  Epoch {epoch:02d} | loss {train_loss:.4f} | "
              f"train {train_acc:.4f} | val {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    return best_val_acc, best_state, epoch, time.time() - t0

# After finding the best hyperparameters on the validation set, we retrain a fresh model on the 
# combined train+val dataset and evaluate it on the test set. This gives us an unbiased estimate 
# of the final test accuracy for the chosen hyperparameters.
def retrain_and_test(model_fn, combined_loader, test_loader, lr, optimizer_name, weight_decay, epochs=10):
    model     = compile_model(model_fn().to(DEVICE))
    criterion = nn.CrossEntropyLoss()
    optimizer = (
        optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adam"
        else optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    )
    t0 = time.time()
    for _ in range(epochs):
        train_epoch(model, combined_loader, criterion, optimizer)
    return evaluate(model, test_loader), time.time() - t0

# Hyperparameter Search Functions
def mlp_search(dataset, depth, configs, get_loaders_fn, input_dim, num_classes, epochs=10):
    
    initial_batch = configs[0]['batch_size']
    get_loaders_fn(batch_size=initial_batch)

    title = f"MLP [{depth.upper()}] on {dataset}"
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")

    best_val, best_cfg = 0.0, None

    # We iterate over the predefined hyperparameter configurations for MLPs, which include different learning rates, 
    # batch sizes, optimizers, and dropout values. For each configuration, we train the model and evaluate it on the 
    # validation set, keeping track of the best validation accuracy and corresponding hyperparameters.
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] lr={cfg['lr']} | batch={cfg['batch_size']} "
            f"| opt={cfg['optimizer']} | dropout={cfg['dropout']}")
        
        train_loader, val_loader, _ = get_loaders_fn(cfg['batch_size'])
        
        model = compile_model(
            build_mlp(depth, input_dim, num_classes, cfg['dropout']).to(DEVICE)
        )
        
        val_acc, _, ep, elapsed = run_training(
            model, train_loader, val_loader, epochs=epochs,
            lr=cfg['lr'], optimizer_name=cfg['optimizer'], patience=3,
        )
    
        print(f"    ✓ val_acc={val_acc:.4f} | epochs={ep} | time={elapsed:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            best_cfg = {**cfg, "val_acc": val_acc}

    print(f"\n  Best val_acc={best_val:.4f} | config={best_cfg}")
    return best_cfg, best_val

# Similar to mlp_search but tailored for CNNs, which use weight decay instead of dropout and typically only use the Adam optimizer.
def cnn_search(dataset, arch, model_fn_factory, get_loaders_fn, configs, epochs=10):
    
    initial_batch = configs[0]['batch_size']
    get_loaders_fn(batch_size=initial_batch)

    title = f"CNN [{arch.upper()}] on {dataset}"
    print(f"\n{'='*60}")
    print(f"{title.center(60)}")
    print(f"{'='*60}")    
    
    best_val, best_cfg = 0.0, None
    
    # We iterate over the predefined hyperparameter configurations for CNNs, which include 
    # different learning rates, batch sizes, and weight decay values. For each configuration, 
    # we train the model and evaluate it on the validation set, keeping track of the best validation accuracy and corresponding hyperparameters.
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] lr={cfg['lr']} | batch={cfg['batch_size']} "
            f"| wd={cfg['weight_decay']}")
            
        train_loader, val_loader, _ = get_loaders_fn(cfg['batch_size'])
            
        model = compile_model(model_fn_factory().to(DEVICE))
            
        val_acc, _, ep, elapsed = run_training(model, train_loader, val_loader, epochs=epochs, 
                lr=cfg['lr'], optimizer_name="adam", weight_decay=cfg['weight_decay'], patience=3,
        )

        print(f"    ✓ val_acc={val_acc:.4f} | epochs={ep} | time={elapsed:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            best_cfg = {**cfg, "val_acc": val_acc}

    print(f"\n  Best val_acc={best_val:.4f} | config={best_cfg}")
    return best_cfg, best_val

# Hyperparameter Grids (10 configs each for MLPs and CNNs, as specified in the project instructions).
MLP_CONFIGS = [
    {"lr": 0.001,  "batch_size": 64,  "optimizer": "adam", "dropout": 0.2},
    {"lr": 0.001,  "batch_size": 128, "optimizer": "adam", "dropout": 0.2},
    {"lr": 0.001,  "batch_size": 256, "optimizer": "adam", "dropout": 0.2},
    {"lr": 0.001,  "batch_size": 64,  "optimizer": "adam", "dropout": 0.5},
    {"lr": 0.01,   "batch_size": 64,  "optimizer": "adam", "dropout": 0.2},
    {"lr": 0.01,   "batch_size": 256, "optimizer": "adam", "dropout": 0.5},
    {"lr": 0.0001, "batch_size": 64,  "optimizer": "adam", "dropout": 0.2},
    {"lr": 0.001,  "batch_size": 64,  "optimizer": "sgd",  "dropout": 0.2},
    {"lr": 0.01,   "batch_size": 64,  "optimizer": "sgd",  "dropout": 0.2},
    {"lr": 0.01,   "batch_size": 256, "optimizer": "sgd",  "dropout": 0.5},
]

CNN_CONFIGS = [
    {"lr": 0.001,  "batch_size": 64,  "weight_decay": 0},
    {"lr": 0.001,  "batch_size": 128, "weight_decay": 1e-4},
    {"lr": 0.001,  "batch_size": 256, "weight_decay": 1e-4},
    {"lr": 0.001,  "batch_size": 64,  "weight_decay": 5e-3},
    {"lr": 0.01,   "batch_size": 64,  "weight_decay": 0},
    {"lr": 0.01,   "batch_size": 256, "weight_decay": 1e-4},
    {"lr": 0.01,   "batch_size": 64,  "weight_decay": 5e-3},
    {"lr": 0.0001, "batch_size": 64,  "weight_decay": 0},
    {"lr": 0.0001, "batch_size": 256, "weight_decay": 1e-4},
    {"lr": 0.01,   "batch_size": 32,  "weight_decay": 1e-4},
]

# Results Saving Utility
SAVE_PATH = "results.json"

# This function saves the results dictionary to a JSON file after each experiment. 
def save_results(results):
    with open(SAVE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved] → {SAVE_PATH}\n")

# Main Experiment Loop
def run_all_experiments():
    results     = {}
    MNIST_DIM   = 28 * 28     
    CIFAR_DIM   = 32 * 32 * 3   
    NUM_CLASSES = 10
    EPOCHS      = 10

    # We first run the hyperparameter search for MLPs on both datasets, iterating over the defined depths 
    # (shallow, medium, deep) and configurations. After finding the best hyperparameters, we retrain a fresh 
    # model on the combined train+val set and evaluate it on the test set, saving the results.
    for depth in ["shallow", "medium", "deep"]:

        # MNIST
        best_cfg, _ = mlp_search("MNIST", depth, MLP_CONFIGS, get_mnist_loaders, MNIST_DIM, NUM_CLASSES, epochs=EPOCHS)
        train_l, val_l, test_l = get_mnist_loaders(best_cfg["batch_size"])
        full_loader = _make_loader(
            ConcatDataset([train_l.dataset, val_l.dataset]),
            best_cfg["batch_size"], shuffle=True,
        )
        test_acc, runtime = retrain_and_test(
            lambda d=depth, dr=best_cfg["dropout"]:
                build_mlp(d, MNIST_DIM, NUM_CLASSES, dr),
            full_loader, test_l,
            lr=best_cfg["lr"], optimizer_name=best_cfg["optimizer"],
            weight_decay=0.0, epochs=EPOCHS,
        )
        results[f"MLP_{depth}_MNIST"] = {
            **best_cfg, "test_acc": test_acc, "runtime": runtime}
        print(f"\n  ★ MNIST MLP [{depth.upper()}] - TEST ACC: {test_acc:.4f} @ {runtime:.1f}s\n")
        save_results(results)

        # CIFAR-10
        best_cfg, _ = mlp_search("CIFAR-10", depth, MLP_CONFIGS, get_cifar10_loaders, CIFAR_DIM, NUM_CLASSES, epochs=EPOCHS)
        train_l, val_l, test_l = get_cifar10_loaders(best_cfg["batch_size"])
        full_loader = _make_loader(
            ConcatDataset([train_l.dataset, val_l.dataset]),
            best_cfg["batch_size"], shuffle=True,
        )
        test_acc, runtime = retrain_and_test(
            lambda d=depth, dr=best_cfg["dropout"]:
                build_mlp(d, CIFAR_DIM, NUM_CLASSES, dr),
            full_loader, test_l,
            lr=best_cfg["lr"], optimizer_name=best_cfg["optimizer"],
            weight_decay=0.0, epochs=EPOCHS,
        )
        results[f"MLP_{depth}_CIFAR10"] = {
            **best_cfg, "test_acc": test_acc, "runtime": runtime}
        print(f"\n  ★ CIFAR-10 MLP [{depth.upper()}] - TEST ACC: {test_acc:.4f} @ {runtime:.1f}s\n")
        save_results(results)

    # Next, we run the hyperparameter search for CNNs on both datasets, iterating over the defined architectures (simple, enhanced) and configurations. 
    # Similar to the MLPs, we retrain a fresh model with the best hyperparameters on the combined train+val set and evaluate it on the test set
    cnn_archs = {
        "simple":   {"mnist": lambda: SimpleCNN(1, 10),
                     "cifar": lambda: SimpleCNN(3, 10)},
        "enhanced": {"mnist": lambda: EnhancedCNN(1, 10),
                     "cifar": lambda: EnhancedCNN(3, 10)},
    }

    for arch, factories in cnn_archs.items():

        # MNIST
        best_cfg, _ = cnn_search("MNIST", arch, factories["mnist"], get_mnist_loaders, CNN_CONFIGS, epochs=EPOCHS)
        train_l, val_l, test_l = get_mnist_loaders(best_cfg["batch_size"])
        full_loader = _make_loader(
            ConcatDataset([train_l.dataset, val_l.dataset]),
            best_cfg["batch_size"], shuffle=True,
        )
        test_acc, runtime = retrain_and_test(
            factories["mnist"], full_loader, test_l,
            lr=best_cfg["lr"], optimizer_name="adam",
            weight_decay=best_cfg["weight_decay"], epochs=EPOCHS,
        )
        results[f"CNN_{arch}_MNIST"] = {
            **best_cfg, "test_acc": test_acc, "runtime": runtime}
        print(f"\n  ★ MNIST CNN [{arch.upper()}] - TEST ACC: {test_acc:.4f} @ {runtime:.1f}s\n")
        save_results(results)

        # CIFAR-10
        best_cfg, _ = cnn_search("CIFAR-10", arch, factories["cifar"], get_cifar10_loaders, CNN_CONFIGS, epochs=EPOCHS)
        train_l, val_l, test_l = get_cifar10_loaders(best_cfg["batch_size"])
        full_loader = _make_loader(
            ConcatDataset([train_l.dataset, val_l.dataset]),
            best_cfg["batch_size"], shuffle=True,
        )
        test_acc, runtime = retrain_and_test(
            factories["cifar"], full_loader, test_l,
            lr=best_cfg["lr"], optimizer_name="adam",
            weight_decay=best_cfg["weight_decay"], epochs=EPOCHS,
        )
        results[f"CNN_{arch}_CIFAR10"] = {
            **best_cfg, "test_acc": test_acc, "runtime": runtime}
        print(f"\n  ★ CIFAR-10 CNN [{arch.upper()}] - TEST ACC: {test_acc:.4f} @ {runtime:.1f}s\n")
        save_results(results)

    return results

# Results Tables
def print_tables(results):
    sep = "-" * 72

    print("\n\n" + "="*72)
    print("TABLE 1: MNIST Results (MLPs)".center(72))
    print("="*72)
    print(f"{'Architecture':<20} {'LR':>7} {'Batch':>6} {'Opt':>5} "
          f"{'Drop':>5} {'TestAcc':>8} {'Runtime':>9}")
    print(sep)
    for depth in ["shallow", "medium", "deep"]:
        r = results.get(f"MLP_{depth}_MNIST", {})
        print(f"{'MLP ('+depth+')':<20} {r.get('lr','-'):>7} "
              f"{r.get('batch_size','-'):>6} {r.get('optimizer','-'):>5} "
              f"{r.get('dropout','-'):>5} {r.get('test_acc',0):>8.4f} "
              f"{r.get('runtime',0):>8.1f}s")

    print("\n\n" + "="*72)
    print("TABLE 2: CIFAR-10 Results (MLPs)".center(72))
    print("="*72)
    print(f"{'Architecture':<20} {'LR':>7} {'Batch':>6} {'Opt':>5} "
          f"{'Drop':>5} {'TestAcc':>8} {'Runtime':>9}")
    print(sep)
    for depth in ["shallow", "medium", "deep"]:
        r = results.get(f"MLP_{depth}_CIFAR10", {})
        print(f"{'MLP ('+depth+')':<20} {r.get('lr','-'):>7} "
              f"{r.get('batch_size','-'):>6} {r.get('optimizer','-'):>5} "
              f"{r.get('dropout','-'):>5} {r.get('test_acc',0):>8.4f} "
              f"{r.get('runtime',0):>8.1f}s")

    print("\n\n" + "="*72)
    print("TABLE 3: MNIST Results (CNNs)".center(72))
    print("="*72)
    print(f"{'Architecture':<22} {'LR':>7} {'Batch':>6} {'WDecay':>8} "
          f"{'TestAcc':>8} {'Runtime':>9}")
    print(sep)
    for arch in ["simple", "enhanced"]:
        r = results.get(f"CNN_{arch}_MNIST", {})
        print(f"{'CNN ('+arch+')':<22} {r.get('lr','-'):>7} "
              f"{r.get('batch_size','-'):>6} {r.get('weight_decay','-'):>8} "
              f"{r.get('test_acc',0):>8.4f} {r.get('runtime',0):>8.1f}s")

    print("\n\n" + "="*72)
    print("TABLE 4: CIFAR-10 Results (CNNs)".center(72))
    print("="*72)
    print(f"{'Architecture':<22} {'LR':>7} {'Batch':>6} {'WDecay':>8} "
          f"{'TestAcc':>8} {'Runtime':>9}")
    print(sep)
    for arch in ["simple", "enhanced"]:
        r = results.get(f"CNN_{arch}_CIFAR10", {})
        print(f"{'CNN ('+arch+')':<22} {r.get('lr','-'):>7} "
              f"{r.get('batch_size','-'):>6} {r.get('weight_decay','-'):>8} "
              f"{r.get('test_acc',0):>8.4f} {r.get('runtime',0):>8.1f}s")

# Entry Point
if __name__ == "__main__":
    results = run_all_experiments()
    print_tables(results)