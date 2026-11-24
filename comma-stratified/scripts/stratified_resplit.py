#!/usr/bin/env python3
"""
Stratified Re-split Script for Balanced Alignment/SLT Distribution

This script:
1. Combines all three CSV files (train/dev/test)
2. Separates by is_alignment_eligible flag
3. Splits each group independently with same ratios
4. Saves new balanced CSV files

Random Seed: 42 (same as bracket format for consistency)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#######################################
# CONFIGURATION
#######################################
SEED = 42  # Same seed as bracket format dataset
TRAIN_RATIO = 0.86
DEV_RATIO = 0.07
TEST_RATIO = 0.07

# File paths
INPUT_DIR = "/home/bpolat/slt_main_2"
OUTPUT_DIR = "/home/bpolat/slt_main_2"

input_files = {
    'train': f"{INPUT_DIR}/sebeder.main.train.paraphrase.alignment.csv",
    'dev': f"{INPUT_DIR}/sebeder.main.dev.paraphrase.alignment.csv",
    'test': f"{INPUT_DIR}/sebeder.main.test.paraphrase.alignment.csv"
}

output_files = {
    'train': f"{OUTPUT_DIR}/sebeder.main.train.paraphrase.alignment.stratified.csv",
    'dev': f"{OUTPUT_DIR}/sebeder.main.dev.paraphrase.alignment.stratified.csv",
    'test': f"{OUTPUT_DIR}/sebeder.main.test.paraphrase.alignment.stratified.csv"
}

def main():
    print("="*60)
    print("Stratified Dataset Re-split (Comma Format)")
    print(f"Random Seed: {SEED}")
    print("="*60)
    
    # Set random seed for reproducibility
    np.random.seed(SEED)
    
    # Step 1: Load and combine all CSVs
    print("\n[Step 1] Loading original CSV files...")
    dfs = []
    for name, path in input_files.items():
        df = pd.read_csv(path)
        df['_original_split'] = name  # Track original split for reference
        dfs.append(df)
        print(f"  {name}: {len(df)} samples")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total combined: {len(combined_df)} samples")
    
    # Step 2: Separate by alignment eligibility
    print("\n[Step 2] Separating by is_alignment_eligible...")
    alignment_df = combined_df[combined_df['is_alignment_eligible'] == True].copy()
    slt_df = combined_df[combined_df['is_alignment_eligible'] == False].copy()
    
    print(f"  Alignment eligible: {len(alignment_df)} samples")
    print(f"  SLT only: {len(slt_df)} samples")
    
    # Step 3: Stratified split for each group
    print("\n[Step 3] Performing stratified splits...")
    
    def split_group(df, group_name):
        # First split: train vs (dev+test)
        train, temp = train_test_split(
            df, 
            train_size=TRAIN_RATIO, 
            random_state=SEED,
            shuffle=True
        )
        
        # Second split: dev vs test (equal split of remainder)
        dev, test = train_test_split(
            temp, 
            train_size=0.5,  # 50% of remainder to each
            random_state=SEED,
            shuffle=True
        )
        
        print(f"  {group_name}: train={len(train)}, dev={len(dev)}, test={len(test)}")
        return train, dev, test
    
    align_train, align_dev, align_test = split_group(alignment_df, "Alignment")
    slt_train, slt_dev, slt_test = split_group(slt_df, "SLT")
    
    # Step 4: Combine groups for final splits
    print("\n[Step 4] Combining groups for final splits...")
    
    final_train = pd.concat([align_train, slt_train], ignore_index=True).sample(frac=1, random_state=SEED)
    final_dev = pd.concat([align_dev, slt_dev], ignore_index=True).sample(frac=1, random_state=SEED)
    final_test = pd.concat([align_test, slt_test], ignore_index=True).sample(frac=1, random_state=SEED)
    
    # Remove tracking column before saving
    for df in [final_train, final_dev, final_test]:
        df.drop('_original_split', axis=1, inplace=True)
    
    # Step 5: Verify and save
    print("\n[Step 5] Verification...")
    print("\n" + "-"*60)
    print("BEFORE (Original):")
    print("-"*60)
    for name, path in input_files.items():
        df = pd.read_csv(path)
        align_pct = df['is_alignment_eligible'].sum() / len(df) * 100
        print(f"  {name}: {len(df)} samples, {align_pct:.1f}% alignment")
    
    print("\n" + "-"*60)
    print("AFTER (Stratified):")
    print("-"*60)
    for name, df in [('train', final_train), ('dev', final_dev), ('test', final_test)]:
        align_pct = df['is_alignment_eligible'].sum() / len(df) * 100
        align_count = df['is_alignment_eligible'].sum()
        print(f"  {name}: {len(df)} samples, {align_count} alignment ({align_pct:.1f}%)")
    
    # Save
    print("\n[Step 6] Saving new CSV files...")
    final_train.to_csv(output_files['train'], index=False)
    final_dev.to_csv(output_files['dev'], index=False)
    final_test.to_csv(output_files['test'], index=False)
    
    print(f"\n  ✅ Saved: {output_files['train']}")
    print(f"  ✅ Saved: {output_files['dev']}")
    print(f"  ✅ Saved: {output_files['test']}")
    
    print("\n" + "="*60)
    print(f"Done! Stratified re-split complete (seed={SEED})")
    print("="*60)

if __name__ == "__main__":
    main()
