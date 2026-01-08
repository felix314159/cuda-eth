// PREREQ:  sudo apt install nvidia-cuda-toolkit libsecp256k1-dev
#include <ctype.h>          // tolower()
#include <stdio.h>
#include <stdint.h>         // stuff like uint32_t
#include <cuda_runtime.h>   // gpu info
#include <secp256k1.h>
#include <chrono>
#include <random>           // random seed generation
#include <cstring>          // strlen

// -------------------- tiny-Keccak implementation ----------------------
#define KECCAK_ROUNDS 24

static const uint64_t keccakf_rndc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

static const int keccakf_rotc[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
};

static const int keccakf_piln[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
};

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

void keccakf(uint64_t st[25]) {
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        uint64_t bc[5];

        for (int i = 0; i < 5; i++)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];

        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        uint64_t t = st[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, keccakf_rotc[i]);
            t = bc[0];
        }

        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        st[0] ^= keccakf_rndc[round];
    }
}

void keccak256(const uint8_t *in, size_t inlen, uint8_t *md) {
    uint64_t st[25] = {0};
    const int rsiz = 136; // 200 - 2 * 32

    for (; inlen >= rsiz; inlen -= rsiz, in += rsiz) {
        for (int i = 0; i < rsiz / 8; i++) {
            st[i] ^= ((uint64_t*)in)[i];
        }
        keccakf(st);
    }

    uint8_t temp[144] = {0};
    for (size_t i = 0; i < inlen; i++)
        temp[i] = in[i];
    temp[inlen] = 0x01;
    temp[rsiz - 1] |= 0x80;

    for (int i = 0; i < rsiz / 8; i++) {
        st[i] ^= ((uint64_t*)temp)[i];
    }
    keccakf(st);

    for (int i = 0; i < 4; i++) {
        ((uint64_t*)md)[i] = st[i];
    }
}

// -------------------- generate 32 bytes (pcg random) ----------------------

__device__ uint32_t pcg_hash(uint32_t input) {
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__global__ void generate_privkeys(unsigned char *output, uint32_t seed, int num_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;

    uint32_t *data = (uint32_t*)&output[idx * 32];
    uint32_t base = (idx * 8) ^ seed;

    for (int i = 0; i < 8; i++) {
        data[i] = pcg_hash(base + i);
    }
}

// ------------- derive Ethereum address from private key ----------------------------
void derive_address(secp256k1_context *ctx, const uint8_t *privkey, uint8_t *address) {
    secp256k1_pubkey pubkey;

    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privkey)) {
        fprintf(stderr, "Failed to create public key\n");
        return;
    }

    uint8_t pubkey_serialized[65];
    size_t pubkey_len = 65;
    secp256k1_ec_pubkey_serialize(ctx, pubkey_serialized, &pubkey_len, &pubkey, SECP256K1_EC_UNCOMPRESSED);

    uint8_t hash[32];
    keccak256(pubkey_serialized + 1, 64, hash);

    for (int i = 0; i < 20; i++) {
        address[i] = hash[12 + i];
    }
}

// ------------- Utility: address -> lowercase hex ----------------------------
void address_to_hex(const uint8_t *address, char *hex_str) {
    for (int i = 0; i < 20; i++) {
        sprintf(hex_str + i * 2, "%02x", address[i]);
    }
    hex_str[40] = '\0';
}

// ------------- New: prefix/suffix matcher (case-insensitive) ----------------
bool matches_prefix_suffix(const char *address_hex, const char *prefix, const char *suffix) {
    const int addr_len = 40;

    const int pre_len = (prefix && *prefix) ? (int)strlen(prefix) : 0;
    const int suf_len = (suffix && *suffix) ? (int)strlen(suffix) : 0;

    if (pre_len > addr_len || suf_len > addr_len) return false;

    if (pre_len) {
        for (int i = 0; i < pre_len; ++i) {
            unsigned char a = (unsigned char)address_hex[i];
            unsigned char b = (unsigned char)prefix[i];
            if (tolower(a) != tolower(b)) return false;
        }
    }

    if (suf_len) {
        const int start = addr_len - suf_len;
        for (int i = 0; i < suf_len; ++i) {
            unsigned char a = (unsigned char)address_hex[start + i];
            unsigned char b = (unsigned char)suffix[i];
            if (tolower(a) != tolower(b)) return false;
        }
    }

    return true;
}

// -------------------- Random seed generation ----------------------
uint32_t get_random_seed() {
    std::random_device rd;
    uint32_t hw_random = rd();

    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    uint32_t time_part = static_cast<uint32_t>(nanos ^ (nanos >> 32));
    uint32_t final_seed = hw_random ^ time_part;

    return final_seed;
}

// -------------------- print gpu info ----------------------
void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // Device 0

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per grid (x,y,z): (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Total CUDA cores (approx): %d\n", prop.multiProcessorCount * 128);  // 128 cores/SM for Ada (approx)
    printf("Maximum concurrent threads: %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("-------------------------------------------------------\n");
}

int main() {
    // ============ CONFIGURATION ============
    const char *prefix = "de";      // address must start with this (case-insensitive),     "" means anything goes
    const char *suffix = "caf";     // address must end with this   (case-insensitive),     "" means anything goes
    // =======================================
    
    int batch_size = 10000; 
    printf("=== Ethereum Vanity Address Miner ===\n");
    print_gpu_info();
    printf("Searching for addresses with");
    if (prefix && *prefix) printf(" prefix '%s'", prefix);
    else                   printf(" no prefix requirement");
    if (suffix && *suffix) printf(" and suffix '%s'", suffix);
    else                   printf(" and no suffix requirement");
    printf("\n");
    printf("Batch size: %d keys\n", batch_size);
    printf("Case-insensitive matching\n");
    uint32_t initial_seed = get_random_seed();
    printf("Initial random seed: 0x%08x\n-------------------------------------------------------", initial_seed);


    secp256k1_context *ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char *d_privkeys;
    cudaMalloc(&d_privkeys, batch_size * 32);
    unsigned char *h_privkeys = new unsigned char[batch_size * 32];
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    uint32_t seed = initial_seed;
    uint64_t total_checked = 0;
    bool found = false;
    auto start_time = std::chrono::high_resolution_clock::now();

    while (!found) {
        generate_privkeys<<<num_blocks, threads_per_block>>>(d_privkeys, seed, batch_size);
        cudaDeviceSynchronize();
        cudaMemcpy(h_privkeys, d_privkeys, batch_size * 32, cudaMemcpyDeviceToHost);

        for (int i = 0; i < batch_size; i++) {
            uint8_t address[20];
            derive_address(ctx, &h_privkeys[i * 32], address);

            char address_hex[41];
            address_to_hex(address, address_hex);

            if (matches_prefix_suffix(address_hex, prefix, suffix)) {
                found = true;

                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
                long secs = duration.count();
                if (secs == 0) secs = 1; // avoid div-by-zero in very fast matches

                printf("\nFOUND MATCHING ADDRESS!\n\n");
                printf("Private Key: 0x");
                for (int j = 0; j < 32; j++) {
                    printf("%02x", h_privkeys[i * 32 + j]);
                }
                printf("\n");
                printf("Address:     0x%s\n", address_hex);
                printf("\nTotal checked: %lu addresses\n", total_checked + i + 1);
                printf("Time elapsed: %ld seconds\n", secs);
                printf("Average speed: %.2f addresses/second\n",
                       (total_checked + i + 1) / (double)secs);
                break;
            }

            total_checked++;

            if (total_checked % 10000 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
                long secs = elapsed.count();
                if (secs == 0) secs = 1;
                double rate = total_checked / (double)secs;
                printf("Checked: %lu addresses | Speed: %.2f addr/s | Time: %lds\r",
                       total_checked, rate, secs);
                fflush(stdout);
            }
        }
        seed = (seed * 1103515245 + 12345) ^ (uint32_t)(total_checked >> 10);
    }

    secp256k1_context_destroy(ctx);
    delete[] h_privkeys;
    cudaFree(d_privkeys);

    return 0;
}

// Compile & Run (optimized for RTX 4070):
//      nvcc -arch=sm_89 -O3 --use_fast_math -Xcompiler -O3 -std=c++11 cuda-miner.cu -o cuda-miner -lsecp256k1 && ./cuda-miner

// Verify with:
//      cast wallet address --private-key <key>

/* Example Output:
=== Ethereum Vanity Address Miner ===
GPU: NVIDIA GeForce RTX 4070
Compute Capability: 8.9
Max threads per block: 1024
Max blocks per grid (x,y,z): (2147483647, 65535, 65535)
Streaming Multiprocessors (SMs): 46
Max threads per SM: 1536
Total CUDA cores (approx): 5888
Maximum concurrent threads: 70656
-------------------------------------------------------
Searching for addresses with prefix 'de' and suffix 'caf'
Batch size: 10000 keys
Case-insensitive matching
Initial random seed: 0xdd054861
Checked: 2870000 addresses | Speed: 71750.00 addr/s | Time: 40s 10000 addresses | Speed: 10000.00 addr/s | Time: 1s
FOUND MATCHING ADDRESS!

Private Key: 0xfb1178767d9b9cc157a7a442e517d0dfddabc4d32929ffc83c1f40fc796a65c4
Address:     0xde5b20df5c96a2b20bc840d9a2c9230681eddcaf

Total checked: 2878321 addresses
Time elapsed: 40 seconds
Average speed: 71958.02 addresses/second
*/
