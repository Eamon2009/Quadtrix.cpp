"""
Export tiktoken GPT-2 vocabulary for C++ inference

This script creates vocabulary files that can be used with C++ tokenizers
like sentencepiece or BPE implementations.
"""

import tiktoken
import json

print("-"*60)
print("             Tokenizer Vocabulary Exporter for C++ Inference              ")
print("-"*60)
print()

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

print(f"Tokenizer: GPT-2 (tiktoken)")
print(f"Vocabulary size: {vocab_size:,}")
print()
#  Export 1: Token to ID mapping (JSON)


print("  → Exporting token-to-id mapping...")

# Get the encoder's vocabulary
# tiktoken doesn't expose the full vocabulary directly, so we reconstruct it
vocab_dict = {}

# Try to decode each token ID to get its string representation
for token_id in range(vocab_size):
    try:
        token_bytes = enc.decode_single_token_bytes(token_id)
        # Store as both hex and string representation
        vocab_dict[token_id] = {
            "hex": token_bytes.hex(),
            "str": token_bytes.decode('utf-8', errors='replace')
        }
    except Exception as e:
        vocab_dict[token_id] = {"hex": "", "str": f"<ERROR_{token_id}>"}

with open('gpt2_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_dict, f, indent=2, ensure_ascii=False)

print(f"Saved: gpt2_vocab.json ({len(vocab_dict)} tokens)")


#  Export 2: Merges file (for BPE)

print("Exporting BPE merges...")

# tiktoken uses a different approach than traditional BPE, but we can try
# to approximate by analyzing common patterns
try:
    # This is a simplified version - for production use, you'd need
    # to extract the actual BPE merge rules from tiktoken
    
    # For now, just create a placeholder
    with open('gpt2_merges.txt', 'w', encoding='utf-8') as f:
        f.write("# GPT-2 BPE Merges (approximation)\n")
        f.write("# For full compatibility, use tiktoken's actual merge rules\n")
    
    print(f"  ⚠ Saved: gpt2_merges.txt (placeholder)")
    print(f"     Note: Full BPE merge extraction requires tiktoken internals")
except Exception as e:
    print(f"  ✗ Could not export merges: {e}")

#  Export 3: Simple byte-pair mapping

print("  Creating token lookup table...")

token_strings = []
for token_id in range(vocab_size):
    try:
        token_bytes = enc.decode_single_token_bytes(token_id)
        token_strings.append(token_bytes.decode('utf-8', errors='replace'))
    except:
        token_strings.append(f"<UNK_{token_id}>")

with open('gpt2_tokens.txt', 'w', encoding='utf-8') as f:
    for token in token_strings:
        # Escape special characters
        escaped = token.replace('\\', '\\\\').replace('\n', '\\n').replace('\t', '\\t')
        f.write(f"{escaped}\n")

print(f" Saved: gpt2_tokens.txt ({len(token_strings)} tokens)")


#  Example encoding/decoding

print()
print("  Testing tokenization...")
test_text = "Hello, how are you?"
test_tokens = enc.encode(test_text)
decoded = enc.decode(test_tokens)

print(f"     Text:    '{test_text}'")
print(f"     Tokens:  {test_tokens}")
print(f"     Decoded: '{decoded}'")


#  Summary
print()
print("Export Complete")
print()
print("Files created:")
print("  gpt2_vocab.json   - Full vocabulary with hex representations")
print(" gpt2_tokens.txt   - Simple token list (one per line)")
print("gpt2_merges.txt   - BPE merges (placeholder)")
print()
print("for C++ integration:")
print("Use sentencepiece with gpt2_vocab.json")
print("Or implement a custom tokenizer using gpt2_tokens.txt")
print("Or use a library like tokenizers (HuggingFace) with C++ bindings")
