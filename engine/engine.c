/*
 *
 * No dependencies 
 * 
 * Compile: gcc -O3 -o gpt_inference chat.c -lm
 * Usage: ./gpt_inference model.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
//  CONFIGURATION (must match Python training config)

#define VOCAB_SIZE 50257  // GPT-2 tokenizer vocabulary size
#define N_EMBD 64
#define N_HEAD 4
#define N_LAYER 4
#define BLOCK_SIZE 32
#define MAX_TOKENS 500
//  DATA STRUCTURES

typedef struct {
    float* data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    float* data;
    int size;
} Vector;

typedef struct {
    // Head components
    Matrix key_weight;
    Matrix query_weight;
    Matrix value_weight;
} Head;

typedef struct {
    Head* heads;
    Matrix proj_weight;
    float* proj_bias;
    int num_heads;
    int head_size;
} MultiHeadAttention;

typedef struct {
    Matrix fc1_weight;
    float* fc1_bias;
    Matrix fc2_weight;
    float* fc2_bias;
} FeedForward;

typedef struct {
    MultiHeadAttention attn;
    FeedForward ffwd;
    float* ln1_weight;
    float* ln1_bias;
    float* ln2_weight;
    float* ln2_bias;
} Block;

typedef struct {
    Matrix token_embedding;
    Matrix position_embedding;
    Block* blocks;
    float* ln_f_weight;
    float* ln_f_bias;
    Matrix lm_head_weight;
    float* lm_head_bias;
} GPTModel;

//  MEMORY MANAGEMENT


Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

Vector create_vector(int size) {
    Vector v;
    v.size = size;
    v.data = (float*)calloc(size, sizeof(float));
    return v;
}

void free_matrix(Matrix* m) {
    if (m->data) free(m->data);
    m->data = NULL;
}

void free_vector(Vector* v) {
    if (v->data) free(v->data);
    v->data = NULL;
}

void matmul(float* out, float* a, float* b, int m, int n, int k) {
    // out(m,k) = a(m,n) @ b(n,k)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float sum = 0.0f;
            for (int l = 0; l < n; l++) {
                sum += a[i * n + l] * b[l * k + j];
            }
            out[i * k + j] = sum;
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void layer_norm(float* out, float* x, float* weight, float* bias, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;
    
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    float std = sqrtf(variance + 1e-5f);
    
    for (int i = 0; i < size; i++) {
        out[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
}

void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
    }
}

// 
//  MODEL OPERATIONS

void attention_head_forward(float* out, float* x, Head* head, int T, int head_size) {
    // Allocate buffers
    float* q = (float*)malloc(T * head_size * sizeof(float));
    float* k = (float*)malloc(T * head_size * sizeof(float));
    float* v = (float*)malloc(T * head_size * sizeof(float));
    float* scores = (float*)malloc(T * T * sizeof(float));
    
    // Compute Q, K, V
    matmul(q, x, head->query_weight.data, T, N_EMBD, head_size);
    matmul(k, x, head->key_weight.data, T, N_EMBD, head_size);
    matmul(v, x, head->value_weight.data, T, N_EMBD, head_size);
    
    // Compute attention scores: Q @ K^T / sqrt(head_size)
    float scale = 1.0f / sqrtf((float)head_size);
    for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
            float sum = 0.0f;
            for (int d = 0; d < head_size; d++) {
                sum += q[i * head_size + d] * k[j * head_size + d];
            }
            scores[i * T + j] = sum * scale;
            
            // Causal mask
            if (j > i) {
                scores[i * T + j] = -INFINITY;
            }
        }
    }
    
    // Apply softmax to each row
    for (int i = 0; i < T; i++) {
        softmax(&scores[i * T], T);
    }
    
    // Apply attention to values
    matmul(out, scores, v, T, T, head_size);
    
    free(q);
    free(k);
    free(v);
    free(scores);
}

void multi_head_attention_forward(float* out, float* x, MultiHeadAttention* mha, int T) {
    int head_size = mha->head_size;
    float* concat = (float*)malloc(T * N_EMBD * sizeof(float));
    float* head_out = (float*)malloc(T * head_size * sizeof(float));
    
    // Run each head
    for (int h = 0; h < mha->num_heads; h++) {
        attention_head_forward(head_out, x, &mha->heads[h], T, head_size);
        
        // Copy to concat buffer
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < head_size; d++) {
                concat[t * N_EMBD + h * head_size + d] = head_out[t * head_size + d];
            }
        }
    }
    
    // Project concatenated heads
    matmul(out, concat, mha->proj_weight.data, T, N_EMBD, N_EMBD);
    
    // Add bias
    for (int t = 0; t < T; t++) {
        for (int d = 0; d < N_EMBD; d++) {
            out[t * N_EMBD + d] += mha->proj_bias[d];
        }
    }
    
    free(concat);
    free(head_out);
}

void feedforward_forward(float* out, float* x, FeedForward* ff, int T) {
    float* hidden = (float*)malloc(T * 4 * N_EMBD * sizeof(float));
    
    // First layer
    matmul(hidden, x, ff->fc1_weight.data, T, N_EMBD, 4 * N_EMBD);
    for (int i = 0; i < T * 4 * N_EMBD; i++) {
        hidden[i] += ff->fc1_bias[i % (4 * N_EMBD)];
    }
    relu(hidden, T * 4 * N_EMBD);
    
    // Second layer
    matmul(out, hidden, ff->fc2_weight.data, T, 4 * N_EMBD, N_EMBD);
    for (int i = 0; i < T * N_EMBD; i++) {
        out[i] += ff->fc2_bias[i % N_EMBD];
    }
    
    free(hidden);
}

void block_forward(float* out, float* x, Block* block, int T) {
    float* attn_out = (float*)malloc(T * N_EMBD * sizeof(float));
    float* ln1_out = (float*)malloc(T * N_EMBD * sizeof(float));
    float* ln2_out = (float*)malloc(T * N_EMBD * sizeof(float));
    float* ff_out = (float*)malloc(T * N_EMBD * sizeof(float));
    
    // Layer norm 1
    for (int t = 0; t < T; t++) {
        layer_norm(&ln1_out[t * N_EMBD], &x[t * N_EMBD], 
                   block->ln1_weight, block->ln1_bias, N_EMBD);
    }
    
    // Attention + residual
    multi_head_attention_forward(attn_out, ln1_out, &block->attn, T);
    for (int i = 0; i < T * N_EMBD; i++) {
        attn_out[i] += x[i];
    }
    
    // Layer norm 2
    for (int t = 0; t < T; t++) {
        layer_norm(&ln2_out[t * N_EMBD], &attn_out[t * N_EMBD], 
                   block->ln2_weight, block->ln2_bias, N_EMBD);
    }
    
    // Feedforward + residual
    feedforward_forward(ff_out, ln2_out, &block->ffwd, T);
    for (int i = 0; i < T * N_EMBD; i++) {
        out[i] = ff_out[i] + attn_out[i];
    }
    
    free(attn_out);
    free(ln1_out);
    free(ln2_out);
    free(ff_out);
}

void gpt_forward(float* logits, GPTModel* model, int* tokens, int T) {
    float* x = (float*)malloc(T * N_EMBD * sizeof(float));
    float* block_out = (float*)malloc(T * N_EMBD * sizeof(float));
    
    // Token + position embeddings
    for (int t = 0; t < T; t++) {
        int tok = tokens[t];
        for (int d = 0; d < N_EMBD; d++) {
            x[t * N_EMBD + d] = model->token_embedding.data[tok * N_EMBD + d] +
                                 model->position_embedding.data[t * N_EMBD + d];
        }
    }
    
    // Run through blocks
    for (int layer = 0; layer < N_LAYER; layer++) {
        block_forward(block_out, x, &model->blocks[layer], T);
        memcpy(x, block_out, T * N_EMBD * sizeof(float));
    }
    
    // Final layer norm
    for (int t = 0; t < T; t++) {
        layer_norm(&block_out[t * N_EMBD], &x[t * N_EMBD], 
                   model->ln_f_weight, model->ln_f_bias, N_EMBD);
    }
    
    // LM head (only compute for last token)
    matmul(logits, &block_out[(T-1) * N_EMBD], model->lm_head_weight.data, 
           1, N_EMBD, VOCAB_SIZE);
    
    if (model->lm_head_bias) {
        for (int i = 0; i < VOCAB_SIZE; i++) {
            logits[i] += model->lm_head_bias[i];
        }
    }
    
    free(x);
    free(block_out);
}

int sample_token(float* logits) {
    softmax(logits, VOCAB_SIZE);
    
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    
    for (int i = 0; i < VOCAB_SIZE; i++) {
        cumsum += logits[i];
        if (r < cumsum) {
            return i;
        }
    }
    
    return VOCAB_SIZE - 1;
}


//  MODEL LOADING


int load_model(GPTModel* model, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open model file %s\n", filename);
        return 0;
    }
    
    // Allocate model components
    model->token_embedding = create_matrix(VOCAB_SIZE, N_EMBD);
    model->position_embedding = create_matrix(BLOCK_SIZE, N_EMBD);
    model->blocks = (Block*)malloc(N_LAYER * sizeof(Block));
    model->ln_f_weight = (float*)malloc(N_EMBD * sizeof(float));
    model->ln_f_bias = (float*)malloc(N_EMBD * sizeof(float));
    model->lm_head_weight = create_matrix(N_EMBD, VOCAB_SIZE);
    model->lm_head_bias = (float*)malloc(VOCAB_SIZE * sizeof(float));
    
    int head_size = N_EMBD / N_HEAD;
    
    // Load embeddings
    fread(model->token_embedding.data, sizeof(float), VOCAB_SIZE * N_EMBD, f);
    fread(model->position_embedding.data, sizeof(float), BLOCK_SIZE * N_EMBD, f);
    
    // Load blocks
    for (int layer = 0; layer < N_LAYER; layer++) {
        Block* block = &model->blocks[layer];
        
        // Multi-head attention
        block->attn.num_heads = N_HEAD;
        block->attn.head_size = head_size;
        block->attn.heads = (Head*)malloc(N_HEAD * sizeof(Head));
        
        for (int h = 0; h < N_HEAD; h++) {
            block->attn.heads[h].key_weight = create_matrix(N_EMBD, head_size);
            block->attn.heads[h].query_weight = create_matrix(N_EMBD, head_size);
            block->attn.heads[h].value_weight = create_matrix(N_EMBD, head_size);
            
            fread(block->attn.heads[h].key_weight.data, sizeof(float), N_EMBD * head_size, f);
            fread(block->attn.heads[h].query_weight.data, sizeof(float), N_EMBD * head_size, f);
            fread(block->attn.heads[h].value_weight.data, sizeof(float), N_EMBD * head_size, f);
        }
        
        block->attn.proj_weight = create_matrix(N_EMBD, N_EMBD);
        block->attn.proj_bias = (float*)malloc(N_EMBD * sizeof(float));
        fread(block->attn.proj_weight.data, sizeof(float), N_EMBD * N_EMBD, f);
        fread(block->attn.proj_bias, sizeof(float), N_EMBD, f);
        
        // Layer norms
        block->ln1_weight = (float*)malloc(N_EMBD * sizeof(float));
        block->ln1_bias = (float*)malloc(N_EMBD * sizeof(float));
        block->ln2_weight = (float*)malloc(N_EMBD * sizeof(float));
        block->ln2_bias = (float*)malloc(N_EMBD * sizeof(float));
        fread(block->ln1_weight, sizeof(float), N_EMBD, f);
        fread(block->ln1_bias, sizeof(float), N_EMBD, f);
        fread(block->ln2_weight, sizeof(float), N_EMBD, f);
        fread(block->ln2_bias, sizeof(float), N_EMBD, f);
        
        // Feedforward
        block->ffwd.fc1_weight = create_matrix(N_EMBD, 4 * N_EMBD);
        block->ffwd.fc1_bias = (float*)malloc(4 * N_EMBD * sizeof(float));
        block->ffwd.fc2_weight = create_matrix(4 * N_EMBD, N_EMBD);
        block->ffwd.fc2_bias = (float*)malloc(N_EMBD * sizeof(float));
        fread(block->ffwd.fc1_weight.data, sizeof(float), N_EMBD * 4 * N_EMBD, f);
        fread(block->ffwd.fc1_bias, sizeof(float), 4 * N_EMBD, f);
        fread(block->ffwd.fc2_weight.data, sizeof(float), 4 * N_EMBD * N_EMBD, f);
        fread(block->ffwd.fc2_bias, sizeof(float), N_EMBD, f);
    }
    
    // Final layer norm and LM head
    fread(model->ln_f_weight, sizeof(float), N_EMBD, f);
    fread(model->ln_f_bias, sizeof(float), N_EMBD, f);
    fread(model->lm_head_weight.data, sizeof(float), N_EMBD * VOCAB_SIZE, f);
    fread(model->lm_head_bias, sizeof(float), VOCAB_SIZE, f);
    
    fclose(f);
    printf(" Model loaded successfully from %s\n", filename);
    return 1;
}


//  MAIN


int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.bin>\n", argv[0]);
        return 1;
    }
    
    srand(time(NULL));
    
    GPTModel model;
    if (!load_model(&model, argv[1])) {
        return 1;
    }
    
    printf("\n------------------------------------------------------------------------------\n");
    printf("  QUADTRIX Engine (Pure C)\n");
    printf("------------------------------------------------------------------------------\n\n");
    
    // Example: Simple generation (you'll need to implement proper tokenization)
    int tokens[BLOCK_SIZE];
    tokens[0] = 198;  // Example token (newline in GPT-2)
    int n_tokens = 1;
    
    printf("Generating text (simplified - add BPE tokenizer for full functionality)...\n\n");
    
    float* logits = (float*)malloc(VOCAB_SIZE * sizeof(float));
    
    for (int i = 0; i < 50; i++) {
        int context_len = n_tokens < BLOCK_SIZE ? n_tokens : BLOCK_SIZE;
        int* context = &tokens[n_tokens - context_len];
        
        gpt_forward(logits, &model, context, context_len);
        int next_token = sample_token(logits);
        
        printf("%d ", next_token);
        fflush(stdout);
        
        if (n_tokens < MAX_TOKENS) {
            tokens[n_tokens++] = next_token;
        } else {
            // Shift tokens left
            memmove(tokens, tokens + 1, (MAX_TOKENS - 1) * sizeof(int));
            tokens[MAX_TOKENS - 1] = next_token;
        }
    }
    
    printf("\n\n");
    free(logits);
    
    printf("------------------------------------------------------------------------------\n");
    
    return 0;
}