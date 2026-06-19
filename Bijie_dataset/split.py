import os
import shutil
import random

# ==========================================
# CONFIGURAZIONE
# ==========================================
BASE_DIR = "/datadrive/landslide/Bijie_dataset"
# Puntiamo direttamente ed esclusivamente alla cartella delle frane
SOURCE_DIR = os.path.join(BASE_DIR, "Bijie-landslide-dataset", "landslide")

# Percentuali (64% Train, 16% Val, 20% Test sul totale)
TEST_RATIO = 0.20         
VAL_RATIO_OF_TRAIN = 0.20 

def setup_directories():
    """Crea la struttura di cartelle di destinazione per train, val e test."""
    for split in ['train', 'val', 'test']:
        for folder in ['image', 'mask', 'dem']:
            os.makedirs(os.path.join(BASE_DIR, split, folder), exist_ok=True)

def copy_files(file_list, split_name):
    """Copia i file dalla cartella landslide alle rispettive cartelle di split."""
    for img_name in file_list:
        # Percorsi sorgente
        src_img = os.path.join(SOURCE_DIR, "image", img_name)
        src_dem = os.path.join(SOURCE_DIR, "dem", img_name)
        src_mask = os.path.join(SOURCE_DIR, "mask", img_name)
        
        # Percorsi destinazione
        dst_img = os.path.join(BASE_DIR, split_name, "image", img_name)
        dst_dem = os.path.join(BASE_DIR, split_name, "dem", img_name)
        dst_mask = os.path.join(BASE_DIR, split_name, "mask", img_name)
        
        # Copia i file
        if os.path.exists(src_img): shutil.copy(src_img, dst_img)
        if os.path.exists(src_dem): shutil.copy(src_dem, dst_dem)
        if os.path.exists(src_mask): shutil.copy(src_mask, dst_mask)

def main():
    print("🚀 Inizio divisione del dataset (SOLO FRANE)...")
    setup_directories()
    
    # Raccogli tutti i nomi dei file dalla cartella image delle frane
    img_dir = os.path.join(SOURCE_DIR, "image")
    if not os.path.exists(img_dir):
        print(f"❌ Errore: Impossibile trovare la cartella {img_dir}. Assicurati che il percorso sia corretto.")
        return

    landslide_imgs = os.listdir(img_dir)
    
    # Mischia i file per avere una distribuzione casuale
    random.seed(42) # Seed fisso per riproducibilità
    random.shuffle(landslide_imgs)
    
    # Calcola le partizioni
    total = len(landslide_imgs)
    num_test = int(total * TEST_RATIO)
    num_initial_train = total - num_test
    num_val = int(num_initial_train * VAL_RATIO_OF_TRAIN)
    num_train = num_initial_train - num_val
    
    # Estrai le fette
    test_ls = landslide_imgs[:num_test]
    val_ls = landslide_imgs[num_test : num_test + num_val]
    train_ls = landslide_imgs[num_test + num_val:]
    
    print(f"📦 Totale immagini elaborate: {total}")
    print(f"   ┣ Train: {len(train_ls)} immagini (~64%)")
    print(f"   ┣ Val:   {len(val_ls)} immagini (~16%)")
    print(f"   ┗ Test:  {len(test_ls)} immagini (20%)")
    
    print("\n⏳ Copia dei file in corso (potrebbe richiedere qualche istante)...")
    
    copy_files(train_ls, "train")
    copy_files(val_ls, "val")
    copy_files(test_ls, "test")
    
    print("✅ Divisione completata con successo! Dataset pronto.")

if __name__ == "__main__":
    main()