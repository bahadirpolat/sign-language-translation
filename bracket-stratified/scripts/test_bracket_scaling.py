#!/usr/bin/env python3
"""
Test the bracket timestamp scaling/unscaling implementation.
"""
import re

# Import the functions from both files
def scale_bracket_timestamps(text, scale_factor=10):
    """
    Scale timestamps in bracket format by multiplying by scale_factor.
    """
    if not text or not isinstance(text, str):
        return text
    
    def scale_match(match):
        val = float(match.group(1))
        scaled = int(round(val * scale_factor))
        return f'[{scaled}]'
    
    pattern = r'\[(\d+\.\d+)\]'
    return re.sub(pattern, scale_match, text)


def unscale_bracket_timestamps(text, scale_factor=10):
    """
    Unscale timestamps in bracket format by dividing by scale_factor.
    """
    if not text or not isinstance(text, str):
        return text
    
    def unscale_match(match):
        val = float(match.group(1))
        unscaled = val / scale_factor
        return f'[{unscaled:.1f}]'
    
    pattern = r'\[(\d+(?:\.\d+)?)\]'
    return re.sub(pattern, unscale_match, text)


def parse_bracket_timestamp_format(text):
    """
    Parse bracket-delimited timestamp format into segments.
    """
    if not text or not isinstance(text, str):
        return []
    
    timestamp_pattern = r'\[(\d+(?:\.\d+)?)\]'
    timestamps = re.findall(timestamp_pattern, text)
    
    if len(timestamps) < 2:
        return []
    
    text_parts = re.split(timestamp_pattern, text)
    
    segments = []
    for i in range(len(timestamps) - 1):
        start = float(timestamps[i])
        end = float(timestamps[i + 1])
        
        text_idx = i * 2 + 2
        if text_idx < len(text_parts):
            segment_text = text_parts[text_idx].strip()
            if segment_text:
                segments.append({
                    'start': start,
                    'end': end,
                    'text': segment_text
                })
    
    return segments


def is_bracket_alignment_format(text):
    """
    Check if text is in bracket timestamp format.
    """
    if not text or not isinstance(text, str):
        return False
    return bool(re.match(r'^\s*\[\d+(?:\.\d+)?\]', text))


print("=" * 60)
print("Bracket Timestamp Scaling Test")
print("=" * 60)

# Test cases from the actual CSV data
test_cases = [
    "[0.0] Peşime taktığın adamların dişlerini söktüm haberin olsun. [2.5] Demek hala yaşıyorsun Ali Yüzbaşı. [4.9]",
    "[0.0] Hadi bakalım Kerem, şimdi bana yardım et ve her zamanki gibi duyduğun her adresi bir yere not etmiş ol. [5.6] Hadi bakalım Kerem, şimdi bana yardım et ve her zamanki gibi duyduğun her adresi bir yere not etmiş ol. [11.3]",
    "[0.0] Konuşmak. [1.5] Seninle konuşacak bir şeyim yok. [3.7]",
]

print("\n[Test 1] Scaling ×10\n")
for i, original in enumerate(test_cases, 1):
    scaled = scale_bracket_timestamps(original)
    print(f"Test {i}:")
    print(f"  Original: {original[:70]}...")
    print(f"  Scaled:   {scaled[:70]}...")
    print()

print("\n[Test 2] Unscaling ÷10\n")
scaled_example = "[0] Konuşmak. [15] Seninle konuşacak bir şeyim yok. [37]"
unscaled = unscale_bracket_timestamps(scaled_example)
print(f"  Scaled:   {scaled_example}")
print(f"  Unscaled: {unscaled}")

print("\n[Test 3] Round-trip (×10 then ÷10)\n")
for i, original in enumerate(test_cases[:1], 1):
    scaled = scale_bracket_timestamps(original)
    unscaled = unscale_bracket_timestamps(scaled)
    print(f"  Original:  {original[:70]}...")
    print(f"  Scaled:    {scaled[:70]}...")
    print(f"  Unscaled:  {unscaled[:70]}...")
    print(f"  Match:     {'✓ PASS' if original == unscaled else '✗ FAIL'}")

print("\n[Test 4] Parsing bracket format\n")
for i, text in enumerate(test_cases[:2], 1):
    segments = parse_bracket_timestamp_format(text)
    print(f"Test {i} - Found {len(segments)} segments:")
    for seg in segments[:2]:  # Show first 2 segments
        print(f"    [{seg['start']:.1f} - {seg['end']:.1f}]: {seg['text'][:40]}...")
    print()

print("\n[Test 5] Alignment format detection\n")
alignment_texts = [
    "[0.0] Hello world [1.5]",
    "[5] Hello [12]",  # Scaled integers
    "Hello world",
    "Merhaba dünya",
    "0,1,hello,1,2,world",  # Old comma format
]
for text in alignment_texts:
    result = is_bracket_alignment_format(text)
    print(f"  {'✓' if result else '✗'} {text[:40]}{'...' if len(text) > 40 else ''}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
