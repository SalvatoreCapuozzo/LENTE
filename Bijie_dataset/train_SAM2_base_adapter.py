import os, glob, json, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from transformers import Sam2Model
import segmentation_models_pytorch as smp

# ==========================================
# 1. CONFIGURAZIONE E PARAMETRI
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
PATIENCE = 15
# ATTENZIONE: SAM 2 Base Plus è molto pesante. Partiamo con Batch = 2.
# Se va in OOM (Out Of Memory), abbassalo a 1.
BATCH_SIZE = 4  
IN_CHANNELS = 4  
DATASET_ROOT = r"/datadrive/landslide/Bijie_dataset"

# --- MODIFICA CHIAVE: Uso di SAM 2 Base Plus dal cloud di Hugging Face ---
SAM_CKPT_PATH = "facebook/sam2-hiera-base-plus"
CHECKPOINT_NAME = "sam2_base_plus_best_landslide_adapter.pth" 

torch.backends.cudnn.benchmark = True

# ==========================================
# 2. DATASET IN-MEMORY (RAM) PER BIJIE
# ==========================================
class BijieLandslideDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.img_dir = os.path.join(root_dir, split, "image")
        self.mask_dir = os.path.join(root_dir, split, "mask")
        self.dem_dir = os.path.join(root_dir, split, "dem")
        
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
        self.transform = transform
        
        print(f"📥 Caricamento dello split '{split}' in Memoria RAM... attendere.")
        self.data = []
        for img_path in tqdm(self.images, desc=f"Loading {split} in RAM"):
            basename = os.path.basename(img_path)
            mask_path = os.path.join(self.mask_dir, basename)
            dem_path = os.path.join(self.dem_dir, basename)
            
            # Carica Immagine RGB (3 canali)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Carica DEM (1 canale) e lo ridimensiona all'ottico
            dem = cv2.imread(dem_path, cv2.IMREAD_GRAYSCALE)
            dem = cv2.resize(dem, (w, h), interpolation=cv2.INTER_LINEAR)
            dem = np.expand_dims(dem, axis=-1) 
            
            # Combina RGB e DEM in un'unica immagine a 4 canali
            combined_img = np.concatenate([image, dem], axis=-1).astype(np.float32) / 255.0
            
            # Carica Mask (1 canale binario)
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.float32)
            else:
                mask = np.zeros((h, w), dtype=np.float32)
                
            self.data.append((combined_img, mask))
        print(f"✅ {len(self.data)} immagini di {split} caricate in RAM!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask)

        indices = torch.where(mask > 0)
        if len(indices[0]) > 0:
            y_min, y_max = indices[0].min().float(), indices[0].max().float()
            x_min, x_max = indices[1].min().float(), indices[1].max().float()
            box = torch.tensor([x_min, y_min, x_max, y_max], dtype=torch.float)
        else:
            box = torch.tensor([0, 0, 10, 10], dtype=torch.float)

        return {"image": image, "mask": mask, "boxes": box}

# ==========================================
# 3. MODELLO SAM 2 ADAPTER (Dinamico per Hiera HF)
# ==========================================
class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(dim, adapter_dim),
            nn.ReLU(inplace=True),
            nn.Linear(adapter_dim, dim)
        )
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)

    def forward(self, x):
        return x + self.adapter(x)

class DSDFEarlyGuidance(nn.Module):
    def __init__(self):
        super().__init__()
        self.guidance_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, terrain):
        return self.guidance_net(terrain)

class SAM2AdapterModel(nn.Module):
    def __init__(self, in_channels=4, adapter_dim=64):
        super().__init__()
        # Usiamo esplicitamente il modello SAM 2 Base Plus da Hugging Face
        self.model = Sam2Model.from_pretrained(SAM_CKPT_PATH)
        self.early_guidance = DSDFEarlyGuidance()
        
        v_encoder = self.model.vision_encoder

        # Iniezione Dinamica degli Adapter tramite Forward Hooks
        self.adapters = nn.ModuleList()
        for name, module in v_encoder.named_modules():
            if hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear):
                dim = module.proj.out_features if hasattr(module, 'proj') else module.qkv.in_features
                new_adapter = Adapter(dim, adapter_dim)
                self.adapters.append(new_adapter)
                module.register_forward_hook(self._get_forward_hook(new_adapter))
                
            elif hasattr(module, 'q_proj') and isinstance(module.q_proj, nn.Linear):
                dim = module.out_proj.out_features if hasattr(module, 'out_proj') else module.q_proj.in_features
                new_adapter = Adapter(dim, adapter_dim)
                self.adapters.append(new_adapter)
                module.register_forward_hook(self._get_forward_hook(new_adapter))

        self._freeze_all_but_adapters()

    def _get_forward_hook(self, adapter):
        def hook(module, args, output):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            else:
                return adapter(output)
        return hook

    def _freeze_all_but_adapters(self):
        # Congela tutto il modello base
        for param in self.model.parameters(): param.requires_grad = False
        
        # Sblocca solo i moduli custom
        for adapter in self.adapters:
            for param in adapter.parameters(): param.requires_grad = True
        for param in self.model.mask_decoder.parameters(): param.requires_grad = True
        for param in self.early_guidance.parameters(): param.requires_grad = True

    def forward(self, x, boxes):
        # Divisione corretta per i 4 canali del Bijie Dataset
        optical = x[:, :3, :, :] 
        terrain = x[:, 3:, :, :] 
        
        g1 = self.early_guidance(terrain)
        
        if boxes.ndim == 2:
            boxes = boxes.unsqueeze(1) 
            
        outputs = self.model(
            pixel_values=optical,
            input_boxes=boxes,
            multimask_output=False
        )
        
        scores = getattr(outputs, "iou_scores", None)
        if scores is None: scores = getattr(outputs, "iou_predictions", None)
        if scores is None: scores = getattr(outputs, "pred_ious", None)
            
        return outputs.pred_masks, scores

# ==========================================
# 4. UTILS & LOSS
# ==========================================
dice_loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

def get_best_mask(all_preds, scores):
    if all_preds.ndim == 5:
        all_preds = all_preds.squeeze(1)
        
    if scores is not None:
        if scores.ndim == 3:
            scores = scores.squeeze(1)
        best_indices = torch.argmax(scores, dim=-1)
        batch_indices = torch.arange(all_preds.shape[0], device=all_preds.device)
        return all_preds[batch_indices, best_indices].unsqueeze(1).contiguous()
    else:
        if all_preds.shape[1] > 1:
            return all_preds[:, 0:1, :, :].contiguous()
        return all_preds.contiguous()

def criterion(y_pred, y_true, lambda_dice=0.3):
    y_pred = y_pred.contiguous()
    y_true_res = F.interpolate(y_true.unsqueeze(1).float(), size=y_pred.shape[2:], mode='nearest')
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true_res)
    d_loss = dice_loss_fn(y_pred, y_true_res)
    return ce_loss + lambda_dice * d_loss

def calculate_metrics(preds, masks):
    if preds.shape[-1] != masks.shape[-1]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear')
    
    pr = (torch.sigmoid(preds) > 0.5).float()
    gt = masks.unsqueeze(1).float()
    
    tp = (pr * gt).sum(dim=(2, 3))
    fp = (pr * (1 - gt)).sum(dim=(2, 3))
    fn = ((1 - pr) * gt).sum(dim=(2, 3))
    
    eps = 1e-7
    
    union = tp + fp + fn
    iou = ((tp + eps) / (union + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    
    return iou, precision, recall

def save_checkpoint(model, optimizer, scheduler, epoch, best_iou, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_iou': best_iou
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0, 0.0
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return checkpoint['epoch'] + 1, checkpoint['best_iou']

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train_model():
    GlobalHydra.instance().clear()
    initialize(config_path=".", job_name="sam2_landslide", version_base=None)
    
    # Aggiornato il nome del file history
    history_file = "training_history_SAM2_base_plus_adapter_metrics.json"
    TARGET_SIZE = 1024  

    train_trans = A.Compose([
        A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
        A.HorizontalFlip(p=0.5), 
        A.RandomRotate90(p=0.5), 
        ToTensorV2()
    ])
    
    val_trans = A.Compose([
        A.Resize(height=TARGET_SIZE, width=TARGET_SIZE),
        ToTensorV2()
    ])

    WORKERS = 0

    train_loader = DataLoader(BijieLandslideDataset(DATASET_ROOT, "train", train_trans), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(BijieLandslideDataset(DATASET_ROOT, "val", val_trans), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)

    model = SAM2AdapterModel().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_iou = load_checkpoint(model, optimizer, scheduler, CHECKPOINT_NAME)
    
    history = []
    if os.path.exists(history_file):
        with open(history_file, "r") as f: history = json.load(f)

    print(f"\n🚀 Inizio addestramento da Epoca {start_epoch+1} su {DEVICE} (SAM 2 Base Plus HF + RAM-Only)")

    patience_counter = 0

    for epoch in range(start_epoch, EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, train_iou_total, train_prec_total, train_rec_total = 0, 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        
        for batch in pbar:
            imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
            
            orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
            imgs = F.interpolate(imgs, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)
            masks = F.interpolate(masks.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode='nearest').squeeze(1)
            
            scale_y, scale_x = TARGET_SIZE / orig_h, TARGET_SIZE / orig_w
            boxes[:, 0] = torch.clamp((boxes[:, 0] * scale_x) - 3, min=0)
            boxes[:, 1] = torch.clamp((boxes[:, 1] * scale_y) - 3, min=0)
            boxes[:, 2] = torch.clamp((boxes[:, 2] * scale_x) + 3, max=TARGET_SIZE-1)
            boxes[:, 3] = torch.clamp((boxes[:, 3] * scale_y) + 3, max=TARGET_SIZE-1)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                all_preds, scores = model(imgs, boxes)
                preds = get_best_mask(all_preds, scores)
                loss = criterion(preds, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            iou, prec, rec = calculate_metrics(preds.detach(), masks)
            train_loss += loss.item()
            train_iou_total += iou
            train_prec_total += prec
            train_rec_total += rec
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}", prec=f"{prec:.4f}", rec=f"{rec:.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_iou_total, val_prec_total, val_rec_total = 0, 0, 0, 0
        with torch.no_grad():
            vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]")
            for batch in vbar:
                imgs, masks, boxes = batch["image"].to(DEVICE), batch["mask"].to(DEVICE), batch["boxes"].to(DEVICE)
                
                orig_h, orig_w = imgs.shape[-2], imgs.shape[-1]
                imgs = F.interpolate(imgs, size=(TARGET_SIZE, TARGET_SIZE), mode='bilinear', align_corners=False)
                masks = F.interpolate(masks.unsqueeze(1), size=(TARGET_SIZE, TARGET_SIZE), mode='nearest').squeeze(1)
                
                scale_y, scale_x = TARGET_SIZE / orig_h, TARGET_SIZE / orig_w
                boxes[:, 0] = torch.clamp((boxes[:, 0] * scale_x) - 3, min=0)
                boxes[:, 1] = torch.clamp((boxes[:, 1] * scale_y) - 3, min=0)
                boxes[:, 2] = torch.clamp((boxes[:, 2] * scale_x) + 3, max=TARGET_SIZE-1)
                boxes[:, 3] = torch.clamp((boxes[:, 3] * scale_y) + 3, max=TARGET_SIZE-1)
                
                with torch.amp.autocast('cuda'):
                    all_preds, scores = model(imgs, boxes)
                    preds = get_best_mask(all_preds, scores)
                    v_loss = criterion(preds, masks).item()
                
                v_iou, v_prec, v_rec = calculate_metrics(preds, masks)
                val_loss += v_loss
                val_iou_total += v_iou
                val_prec_total += v_prec
                val_rec_total += v_rec
                
                vbar.set_postfix(iou=f"{v_iou:.4f}", prec=f"{v_prec:.4f}", rec=f"{v_rec:.4f}")
        
        # --- CALCOLO MEDIE ---
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou_total / len(train_loader)
        avg_train_prec = train_prec_total / len(train_loader)
        avg_train_rec = train_rec_total / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou_total / len(val_loader)
        avg_val_prec = val_prec_total / len(val_loader)
        avg_val_rec = val_rec_total / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        print(f"📊 EPOCA {epoch+1} | Loss T: {avg_train_loss:.4f} | IoU T: {avg_train_iou:.4f} | Prec T: {avg_train_prec:.4f} | Rec T: {avg_train_rec:.4f}")
        print(f"                 | Loss V: {avg_val_loss:.4f} | IoU V: {avg_val_iou:.4f} | Prec V: {avg_val_prec:.4f} | Rec V: {avg_val_rec:.4f}")

        stats = {
            "epoch": epoch+1, 
            "train_loss": avg_train_loss, "train_iou": avg_train_iou, "train_precision": avg_train_prec, "train_recall": avg_train_rec,
            "val_loss": avg_val_loss, "val_iou": avg_val_iou, "val_precision": avg_val_prec, "val_recall": avg_val_rec
        }
        history.append(stats)
        with open(history_file, "w") as f: json.dump(history, f, indent=4)

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            save_checkpoint(model, optimizer, scheduler, epoch, best_iou, CHECKPOINT_NAME)
            print(f"⭐ Nuovo Record IoU: {best_iou:.4f} (Prec: {avg_val_prec:.4f}, Rec: {avg_val_rec:.4f})! - Salvataggio Checkpoint")
            patience_counter = 0 
        else:
            patience_counter += 1
            print(f"⚠️ Nessun miglioramento per l'IoU. Patience: {patience_counter}/{PATIENCE}")
            
            if patience_counter >= PATIENCE:
                print(f"\n🛑 EARLY STOPPING ATTIVATO: La metrica di validazione non migliora da {PATIENCE} epoche.")
                print(f"L'addestramento è stato interrotto per prevenire l'overfitting. Miglior IoU raggiunto: {best_iou:.4f}")
                break

if __name__ == "__main__":
    train_model()