# Copy Fast GMVAE Files to Cloud Instance

## Problem

The fast GMVAE scripts were created locally but you're running on a cloud instance at `/pv2/`.

## Solution: Copy Files

### Option 1: Using Git (Recommended)

```bash
# On local machine (in /Users/sharatsakamuri/Documents/pv2)
git add scripts_v2/2_train_gmvae_fast.py
git add scripts_v2/2_train_gmvae_fast_ddp.py
git commit -m "Add fast GMVAE training scripts"
git push

# On cloud instance (in /pv2)
git pull
```

### Option 2: Using SCP

```bash
# From local machine, copy to cloud
scp scripts_v2/2_train_gmvae_fast.py YOUR_CLOUD:/pv2/scripts_v2/
scp scripts_v2/2_train_gmvae_fast_ddp.py YOUR_CLOUD:/pv2/scripts_v2/
```

### Option 3: Manual Copy-Paste

On cloud instance, create the files:

**File 1: `/pv2/scripts_v2/2_train_gmvae_fast.py`**

Copy entire content from local file, or run:

```bash
# On cloud instance
cat > /pv2/scripts_v2/2_train_gmvae_fast.py << 'EOF'
# [paste entire content here]
EOF
```

**File 2: `/pv2/scripts_v2/2_train_gmvae_fast_ddp.py`**

Same process.

---

## Quick Git Commands

If you have git set up:

```bash
# On local machine
cd /Users/sharatsakamuri/Documents/pv2
git add scripts_v2/2_train_gmvae_fast.py scripts_v2/2_train_gmvae_fast_ddp.py
git commit -m "Add fast GMVAE scripts - 20x speedup"
git push origin main

# On cloud instance
cd /pv2
git pull origin main
```

Then run:

```bash
torchrun --nproc_per_node=2 scripts_v2/2_train_gmvae_fast_ddp.py \
    --data_dir processed_data \
    --output models/gmvae/gmvae_fast.pt \
    --epochs 10 \
    --batch_size 4096 \
    --learning_rate 1e-3
```

---

## Verify Files Copied

On cloud instance:

```bash
ls -lh /pv2/scripts_v2/2_train_gmvae_fast*.py

# Should see:
# -rw-r--r-- ... 2_train_gmvae_fast.py
# -rw-r--r-- ... 2_train_gmvae_fast_ddp.py
```

Make executable:

```bash
chmod +x /pv2/scripts_v2/2_train_gmvae_fast*.py
```
