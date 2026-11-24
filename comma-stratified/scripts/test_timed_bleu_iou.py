#!/usr/bin/env python3
"""
Test script to verify Timed BLEU and IoU metrics implementation.
"""

import sys
sys.path.insert(0, '/home/bpolat/slt_main_2/signpiece-stable')

import numpy as np
from signpiece.utils.training_utils import (
    parse_timestamp_format,
    assign_token_timestamps,
    align_tokens_temporally,
    compute_timed_bleu,
    compute_segment_iou
)

def test_parse_timestamp_format():
    print("\n" + "="*60)
    print("Testing parse_timestamp_format()")
    print("="*60)
    
    # Test case from user's example
    text = "0,1,hello,1,2,world,2,3,bye"
    result = parse_timestamp_format(text)
    
    print(f"Input: {text}")
    print(f"Output: {result}")
    
    expected = [
        {'start': 0.0, 'end': 1.0, 'text': 'hello'},
        {'start': 1.0, 'end': 2.0, 'text': 'world'},
        {'start': 2.0, 'end': 3.0, 'text': 'bye'}
    ]
    
    assert len(result) == 3, f"Expected 3 segments, got {len(result)}"
    print("✓ Correct number of segments")
    
    for i, (res, exp) in enumerate(zip(result, expected)):
        assert res['start'] == exp['start'], f"Segment {i}: wrong start"
        assert res['end'] == exp['end'], f"Segment {i}: wrong end"
        assert res['text'] == exp['text'], f"Segment {i}: wrong text"
    print("✓ All segments parsed correctly")


def test_assign_token_timestamps():
    print("\n" + "="*60)
    print("Testing assign_token_timestamps()")
    print("="*60)
    
    segments = [
        {'start': 0.0, 'end': 2.0, 'text': 'hello world'}
    ]
    
    result = assign_token_timestamps(segments)
    print(f"Input segments: {segments}")
    print(f"Output tokens: {result}")
    
    # Expected: token[0] "hello": 0.0 + 0 × (2.0-0.0)/2 = 0.0
    #           token[1] "world": 0.0 + 1 × (2.0-0.0)/2 = 1.0
    assert len(result) == 2, f"Expected 2 tokens, got {len(result)}"
    assert result[0]['token'] == 'hello', f"Wrong token: {result[0]['token']}"
    assert result[0]['time'] == 0.0, f"Wrong time: {result[0]['time']}"
    assert result[1]['token'] == 'world', f"Wrong token: {result[1]['token']}"
    assert result[1]['time'] == 1.0, f"Wrong time: {result[1]['time']}"
    print("✓ Token timestamps assigned correctly")


def test_align_tokens_temporally():
    print("\n" + "="*60)
    print("Testing align_tokens_temporally()")
    print("="*60)
    
    pred_tokens = [
        {'token': 'hi', 'time': 0.5},
        {'token': 'bye', 'time': 1.5}
    ]
    ref_segments = [
        {'start': 0, 'end': 1, 'text': 'hello'},
        {'start': 1, 'end': 2, 'text': 'world'}
    ]
    
    result = align_tokens_temporally(pred_tokens, ref_segments)
    print(f"Pred tokens: {pred_tokens}")
    print(f"Ref segments: {ref_segments}")
    print(f"Aligned groups: {result}")
    
    # Expected: "hi" (time 0.5) → segment [0,1] "hello"
    #           "bye" (time 1.5) → segment [1,2] "world"
    assert 'hi' in result[0]['pred_tokens'], "Token 'hi' not aligned to segment 0"
    assert 'bye' in result[1]['pred_tokens'], "Token 'bye' not aligned to segment 1"
    print("✓ Tokens aligned correctly to reference segments")


def test_compute_timed_bleu():
    print("\n" + "="*60)
    print("Testing compute_timed_bleu()")
    print("="*60)
    
    # Perfect match case with multi-word segments (better for BLEU-2)
    pred = "0,2,hello world,2,4,goodbye moon"
    ref = "0,2,hello world,2,4,goodbye moon"
    score = compute_timed_bleu(pred, ref)
    print(f"Perfect match (multi-word) - Score: {score:.2f}")
    assert score > 50, f"Expected high score for perfect match, got {score}"
    print("✓ Perfect match gives high score")
    
    # Partial match case
    pred = "0,1,hi,1,2,bye"
    ref = "0,1,hello,1,2,world"
    score = compute_timed_bleu(pred, ref)
    print(f"Partial match (single-word) - Score: {score:.2f}")
    assert 0 <= score <= 100, f"Score out of range: {score}"
    print("✓ Partial match gives reasonable score")
    
    # Similar words case
    pred = "0,2,hello there world"
    ref = "0,2,hello beautiful world"
    score = compute_timed_bleu(pred, ref)
    print(f"Similar words - Score: {score:.2f}")
    assert score > 0, f"Expected positive score for similar words, got {score}"
    print("✓ Similar words case works")


def test_compute_segment_iou():
    print("\n" + "="*60)
    print("Testing compute_segment_iou()")
    print("="*60)
    
    # Perfect overlap case
    pred = "0,1,hello,1,2,world"
    ref = "0,1,hello,1,2,world"
    iou = compute_segment_iou(pred, ref)
    print(f"Perfect overlap - IoU: {iou:.2f}")
    assert iou == 100.0, f"Expected 100% IoU for perfect match, got {iou}"
    print("✓ Perfect overlap gives IoU = 100%")
    
    # 50% overlap case
    pred = "0,2,hello"
    ref = "1,3,hello"
    iou = compute_segment_iou(pred, ref)
    print(f"Partial overlap - IoU: {iou:.2f}")
    # intersection = max(0, min(2, 3) - max(0, 1)) = 1
    # union = max(2, 3) - min(0, 1) = 3
    # iou = 1/3 ≈ 33.33%
    assert 30 <= iou <= 40, f"Expected ~33% IoU, got {iou}"
    print("✓ Partial overlap gives correct IoU")


def main():
    print("\n" + "#"*60)
    print("# Timed BLEU and IoU Metrics Verification Test")
    print("#"*60)
    
    try:
        test_parse_timestamp_format()
        test_assign_token_timestamps()
        test_align_tokens_temporally()
        test_compute_timed_bleu()
        test_compute_segment_iou()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
