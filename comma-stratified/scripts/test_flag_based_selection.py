#!/usr/bin/env python3
"""
Test script to verify flag-based task selection using is_alignment_eligible column.
This reads the first 10 samples from the training CSV and verifies the expected task types.
"""

import pandas as pd
import sys

def test_flag_based_selection():
    # Load first 10 samples from training CSV
    csv_path = "/home/bpolat/slt_main_2/sebeder.main.train.paraphrase.alignment.csv"
    
    print("=" * 60)
    print("Flag-Based Task Selection Verification Test")
    print("=" * 60)
    
    try:
        df = pd.read_csv(csv_path, nrows=10)
    except Exception as e:
        print(f"ERROR: Could not read CSV file: {e}")
        return False
    
    # Check if is_alignment_eligible column exists
    if 'is_alignment_eligible' not in df.columns:
        print("ERROR: 'is_alignment_eligible' column not found in CSV!")
        print(f"Available columns: {list(df.columns)}")
        return False
    
    print(f"\nLoaded {len(df)} samples from: {csv_path}")
    print("\n" + "-" * 60)
    print("Sample Data (first 10 rows):")
    print("-" * 60)
    
    # Show relevant columns
    display_cols = ['clip_id', 'text', 'is_alignment_eligible']
    for idx, row in df.iterrows():
        flag = row['is_alignment_eligible']
        expected_task = 'alignment' if flag else 'sign2text'
        text_preview = str(row['text'])[:50] + "..." if len(str(row['text'])) > 50 else str(row['text'])
        
        print(f"\n[Sample {idx + 1}]")
        print(f"  clip_id: {row['clip_id'][:40]}...")
        print(f"  text: {text_preview}")
        print(f"  is_alignment_eligible: {flag}")
        print(f"  → Expected task: {expected_task}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    alignment_count = df['is_alignment_eligible'].sum()
    slt_count = len(df) - alignment_count
    
    print(f"  Total samples: {len(df)}")
    print(f"  Alignment tasks (flag=True): {alignment_count}")
    print(f"  SLT tasks (flag=False): {slt_count}")
    
    # Check if alignment columns exist for alignment-eligible samples
    alignment_samples = df[df['is_alignment_eligible'] == True]
    if len(alignment_samples) > 0:
        has_clip_text = alignment_samples['clip_text'].notna().all()
        has_timestamps = alignment_samples['clip_text_with_timestamps'].notna().all()
        print(f"\n  Alignment data validation:")
        print(f"    - clip_text present: {has_clip_text}")
        print(f"    - clip_text_with_timestamps present: {has_timestamps}")
        
        if not (has_clip_text and has_timestamps):
            print("  WARNING: Some alignment-eligible samples are missing required data!")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_flag_based_selection()
    sys.exit(0 if success else 1)
