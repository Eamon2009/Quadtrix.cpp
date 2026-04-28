import torch
import sys
import os
import numpy as np

def convert_model_to_bin(pt_path, bin_path):
    """Convert PyTorch .pt file to binary format for C inference"""

    print(f"Loading PyTorch model from {pt_path}...")
    state_dict = torch.load(pt_path, map_location='cpu')

    print("Converting to binary format...")

    with open(bin_path, 'wb') as f:
        # Write embeddings
        print("  - Token embeddings")
        token_emb = state_dict['token_embedding_table.weight'].numpy().flatten()
        f.write(token_emb.tobytes())

        print("  - Position embeddings")
        pos_emb = state_dict['position_embedding_table.weight'].numpy().flatten()
        f.write(pos_emb.tobytes())

        # Model config (must match training)
        n_layer = 4
        n_head = 4
        n_embd = 64

        for layer in range(n_layer):
            print(f"  - Block {layer}")

            for h in range(n_head):
                # Key
                key_weight = state_dict[f'blocks.{layer}.sa.heads.{h}.key.weight'].numpy()
                f.write(key_weight.tobytes())

                # Query
                query_weight = state_dict[f'blocks.{layer}.sa.heads.{h}.query.weight'].numpy()
                f.write(query_weight.tobytes())

                # Value
                value_weight = state_dict[f'blocks.{layer}.sa.heads.{h}.value.weight'].numpy()
                f.write(value_weight.tobytes())

            # Projection
            proj_weight = state_dict[f'blocks.{layer}.sa.proj.weight'].numpy()
            f.write(proj_weight.tobytes())

            proj_bias = state_dict[f'blocks.{layer}.sa.proj.bias'].numpy()
            f.write(proj_bias.tobytes())

            # LayerNorm 1
            ln1_weight = state_dict[f'blocks.{layer}.ln1.weight'].numpy()
            f.write(ln1_weight.tobytes())

            ln1_bias = state_dict[f'blocks.{layer}.ln1.bias'].numpy()
            f.write(ln1_bias.tobytes())

            # LayerNorm 2
            ln2_weight = state_dict[f'blocks.{layer}.ln2.weight'].numpy()
            f.write(ln2_weight.tobytes())

            ln2_bias = state_dict[f'blocks.{layer}.ln2.bias'].numpy()
            f.write(ln2_bias.tobytes())

            # Feedforward
            ffwd_fc1_weight = state_dict[f'blocks.{layer}.ffwd.net.0.weight'].numpy()
            f.write(ffwd_fc1_weight.tobytes())

            ffwd_fc1_bias = state_dict[f'blocks.{layer}.ffwd.net.0.bias'].numpy()
            f.write(ffwd_fc1_bias.tobytes())

            ffwd_fc2_weight = state_dict[f'blocks.{layer}.ffwd.net.2.weight'].numpy()
            f.write(ffwd_fc2_weight.tobytes())

            ffwd_fc2_bias = state_dict[f'blocks.{layer}.ffwd.net.2.bias'].numpy()
            f.write(ffwd_fc2_bias.tobytes())

        # Final LayerNorm
        print("  - Final layer norm")
        ln_f_weight = state_dict['ln_f.weight'].numpy()
        f.write(ln_f_weight.tobytes())

        ln_f_bias = state_dict['ln_f.bias'].numpy()
        f.write(ln_f_bias.tobytes())

        # LM Head
        print("  - Language model head")
        lm_head_weight = state_dict['lm_head.weight'].numpy()
        f.write(lm_head_weight.tobytes())

        if 'lm_head.bias' in state_dict:
            lm_head_bias = state_dict['lm_head.bias'].numpy()
            f.write(lm_head_bias.tobytes())
        else:
            vocab_size = 50257
            zeros = np.zeros(vocab_size, dtype=np.float32)
            f.write(zeros.tobytes())

    print(f"\n Conversion complete!")

    # File info
    abs_path = os.path.abspath(bin_path)
    size_mb = os.path.getsize(bin_path) / (1024 * 1024)

    print(f" Saved to: {abs_path}")
    print(f" File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_pt_to_bin.py <input.pt> <output.bin>")
        sys.exit(1)

    pt_path = sys.argv[1]

    # FORCE OUTPUT DIRECTORY
    output_dir = r"C:\Users\Admin\Documents\GitHub\Quadtrix.cpp\GPU & CPU"
    os.makedirs(output_dir, exist_ok=True)

    # Keep filename but override location
    output_filename = os.path.basename(sys.argv[2])
    bin_path = os.path.join(output_dir, output_filename)

    try:
        convert_model_to_bin(pt_path, bin_path)
    except Exception as e:
        print(f"\n Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)