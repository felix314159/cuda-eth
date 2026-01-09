// PREREQ:  sudo apt install nvidia-cuda-toolkit
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdlib>
#include <cmath>

#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

// CREATE2 inputs (set at runtime via cudaMemcpyToSymbol)
__device__ __constant__ uint8_t kDeployerDev[20];
__device__ __constant__ uint8_t kInitCodeHashDev[32];
__device__ __constant__ uint8_t kBaseSaltDev[32];

// -------------------- tiny-Keccak implementation ----------------------
#define KECCAK_ROUNDS 24

#define KECCAK_RNDC_VALUES \
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, \
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL, \
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL, \
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL, \
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, \
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL, \
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL, \
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL

#define KECCAK_ROTC_VALUES \
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, \
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44

#define KECCAK_PILN_VALUES \
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, \
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1

#if defined(__CUDACC__)
__device__ __constant__ uint64_t keccakf_rndc_dev[24] = { KECCAK_RNDC_VALUES };
__device__ __constant__ int keccakf_rotc_dev[24] = { KECCAK_ROTC_VALUES };
__device__ __constant__ int keccakf_piln_dev[24] = { KECCAK_PILN_VALUES };
#endif

static const uint64_t keccakf_rndc_host[24] = {
    KECCAK_RNDC_VALUES
};

static const int keccakf_rotc_host[24] = {
    KECCAK_ROTC_VALUES
};

static const int keccakf_piln_host[24] = {
    KECCAK_PILN_VALUES
};

#if defined(__CUDA_ARCH__)
#define KECCAK_RNDC keccakf_rndc_dev
#define KECCAK_ROTC keccakf_rotc_dev
#define KECCAK_PILN keccakf_piln_dev
#else
#define KECCAK_RNDC keccakf_rndc_host
#define KECCAK_ROTC keccakf_rotc_host
#define KECCAK_PILN keccakf_piln_host
#endif

#define ROTL64(x, y) (((x) << (y)) | ((x) >> (64 - (y))))

HD uint64_t load64(const uint8_t *x) {
    return ((uint64_t)x[0]) |
           ((uint64_t)x[1] << 8) |
           ((uint64_t)x[2] << 16) |
           ((uint64_t)x[3] << 24) |
           ((uint64_t)x[4] << 32) |
           ((uint64_t)x[5] << 40) |
           ((uint64_t)x[6] << 48) |
           ((uint64_t)x[7] << 56);
}

HD void store64(uint8_t *out, uint64_t v) {
    out[0] = (uint8_t)(v);
    out[1] = (uint8_t)(v >> 8);
    out[2] = (uint8_t)(v >> 16);
    out[3] = (uint8_t)(v >> 24);
    out[4] = (uint8_t)(v >> 32);
    out[5] = (uint8_t)(v >> 40);
    out[6] = (uint8_t)(v >> 48);
    out[7] = (uint8_t)(v >> 56);
}

HD void keccakf(uint64_t st[25]) {
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
            int j = KECCAK_PILN[i];
            bc[0] = st[j];
            st[j] = ROTL64(t, KECCAK_ROTC[i]);
            t = bc[0];
        }

        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; i++)
                bc[i] = st[j + i];
            for (int i = 0; i < 5; i++)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        st[0] ^= KECCAK_RNDC[round];
    }
}

HD void keccak256(const uint8_t *in, size_t inlen, uint8_t *md) {
    uint64_t st[25] = {0};
    const int rsiz = 136; // 200 - 2 * 32

    for (; inlen >= (size_t)rsiz; inlen -= rsiz, in += rsiz) {
        for (int i = 0; i < rsiz / 8; i++) {
            st[i] ^= load64(in + (i * 8));
        }
        keccakf(st);
    }

    uint8_t temp[144] = {0};
    for (size_t i = 0; i < inlen; i++)
        temp[i] = in[i];
    temp[inlen] = 0x01;
    temp[rsiz - 1] |= 0x80;

    for (int i = 0; i < rsiz / 8; i++) {
        st[i] ^= load64(temp + (i * 8));
    }
    keccakf(st);

    for (int i = 0; i < 4; i++) {
        store64(md + (i * 8), st[i]);
    }
}

HD int count_leading_zero_nibbles(const uint8_t *data, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        uint8_t b = data[i];
        if (b == 0) {
            count += 2;
            continue;
        }
        if ((b >> 4) == 0) count += 1;
        break;
    }
    return count;
}

HD void add_offset_to_salt_be(const uint8_t *base, uint64_t offset, uint8_t *out) {
    for (int i = 0; i < 32; i++) {
        out[i] = base[i];
    }
    uint64_t carry = offset;
    for (int i = 31; i >= 0; i--) {
        if (carry == 0) break;
        uint64_t sum = (uint64_t)out[i] + (carry & 0xFFu);
        out[i] = (uint8_t)sum;
        carry = (carry >> 8) + (sum >> 8);
    }
}

__global__ void search_candidates(uint8_t *out_zeroes, uint64_t *out_offsets,
                                  uint64_t seed, int num_candidates, uint64_t nonce) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    uint64_t offset = seed + nonce + (uint64_t)idx;
    uint8_t salt[32];
    add_offset_to_salt_be(kBaseSaltDev, offset, salt);

    uint8_t preimage[85];
    preimage[0] = 0xff;
    for (int i = 0; i < 20; i++) preimage[1 + i] = kDeployerDev[i];
    for (int i = 0; i < 32; i++) preimage[21 + i] = salt[i];
    for (int i = 0; i < 32; i++) preimage[53 + i] = kInitCodeHashDev[i];

    uint8_t hash[32];
    keccak256(preimage, sizeof(preimage), hash);

    out_zeroes[idx] = (uint8_t)count_leading_zero_nibbles(hash + 12, 20);
    out_offsets[idx] = offset;
}

static void bytes_to_hex(const uint8_t *data, size_t len, char *hex_str) {
    for (size_t i = 0; i < len; i++) {
        sprintf(hex_str + i * 2, "%02x", data[i]);
    }
    hex_str[len * 2] = '\0';
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static size_t hex_len_no_prefix(const char *hex) {
    if (hex == nullptr) return 0;
    const char *ptr = hex;
    size_t len = strlen(ptr);
    if (len >= 2 && ptr[0] == '0' && (ptr[1] == 'x' || ptr[1] == 'X')) {
        ptr += 2;
        len -= 2;
    }
    return len;
}

static bool hex_to_bytes_fixed(const char *hex, uint8_t *out, size_t out_len) {
    if (hex == nullptr) return false;
    const char *ptr = hex;
    size_t len = strlen(ptr);
    if (len >= 2 && ptr[0] == '0' && (ptr[1] == 'x' || ptr[1] == 'X')) {
        ptr += 2;
        len -= 2;
    }
    if (len != out_len * 2) return false;
    for (size_t i = 0; i < out_len; i++) {
        int hi = hex_value(ptr[i * 2]);
        int lo = hex_value(ptr[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static bool hex_to_bytes_alloc(const char *hex, uint8_t **out, size_t *out_len) {
    if (hex == nullptr || out == nullptr || out_len == nullptr) return false;
    const char *ptr = hex;
    size_t len = strlen(ptr);
    if (len >= 2 && ptr[0] == '0' && (ptr[1] == 'x' || ptr[1] == 'X')) {
        ptr += 2;
        len -= 2;
    }
    if (len == 0) {
        *out = nullptr;
        *out_len = 0;
        return true;
    }
    if ((len % 2) != 0) return false;
    *out_len = len / 2;
    *out = new uint8_t[*out_len];
    for (size_t i = 0; i < *out_len; i++) {
        int hi = hex_value(ptr[i * 2]);
        int lo = hex_value(ptr[i * 2 + 1]);
        if (hi < 0 || lo < 0) {
            delete[] *out;
            *out = nullptr;
            *out_len = 0;
            return false;
        }
        (*out)[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void fill_random_bytes(uint8_t *out, size_t len, uint64_t seed) {
    uint64_t state = seed;
    size_t i = 0;
    while (i < len) {
        uint64_t r = splitmix64(&state);
        for (int b = 0; b < 8 && i < len; b++, i++) {
            out[i] = (uint8_t)(r >> (8 * b));
        }
    }
}

static void compute_create2_address(const uint8_t deployer[20], const uint8_t salt[32],
                                    const uint8_t init_code_hash[32], uint8_t out_address[20]) {
    uint8_t preimage[85];
    // EIP-1014: keccak256(0xff ++ deployer ++ salt ++ keccak256(init_code))[12:]
    preimage[0] = 0xff;
    for (int i = 0; i < 20; i++) preimage[1 + i] = deployer[i];
    for (int i = 0; i < 32; i++) preimage[21 + i] = salt[i];
    for (int i = 0; i < 32; i++) preimage[53 + i] = init_code_hash[i];

    uint8_t hash[32];
    keccak256(preimage, sizeof(preimage), hash);
    memcpy(out_address, hash + 12, 20);
}

// -------------------- Random seed generation ----------------------
uint64_t get_random_seed() {
    std::random_device rd;
    uint64_t hw_random = ((uint64_t)rd() << 32) ^ (uint64_t)rd();

    auto now = std::chrono::high_resolution_clock::now();
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    uint64_t time_part = (uint64_t)nanos ^ ((uint64_t)nanos >> 32);
    uint64_t final_seed = hw_random ^ time_part;

    return final_seed;
}

static void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
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
    printf("Maximum concurrent threads: %d\n\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
}

int main() {
    // your contracts initcode
    const char *init_code_hex_str = "0x606162";
    
    const int target_zeroes = 10;   // i usually stop mining at 8, i am poor. goal of this program: find this many leading zeroes in hex address
    int report_zeroes = 6;          // results that lead to report_zeroes or more hex leading zeroes are reported in terminal
    
    // ---------------------------------------------
    // nick johnson's create2 factory (the 'to' field of your tx, you pass the calldata to it as 0x + salt + initcode)
    const char *deployer_hex_str = "0x4e59b44847b379578588920cA78FbF26c0B4956C"; // must be 40 chars (20 bytes)

    if (target_zeroes < 1 || target_zeroes > 40) {
        fprintf(stderr, "TARGET_ZEROES must be between 1 and 40.\n");
        return 1;
    }
    if (report_zeroes < 1) report_zeroes = 1;
    if (report_zeroes > target_zeroes) report_zeroes = target_zeroes;

    if (hex_len_no_prefix(deployer_hex_str) != 40) {
        fprintf(stderr, "DEPLOYER must be 40 hex chars (20 bytes).\n");
        return 1;
    }
    uint8_t deployer[20];
    uint8_t base_salt[32];
    if (!hex_to_bytes_fixed(deployer_hex_str, deployer, sizeof(deployer))) {
        fprintf(stderr, "Invalid DEPLOYER hex string (non-hex characters).\n");
        return 1;
    }
    uint64_t salt_seed = get_random_seed();
    fill_random_bytes(base_salt, sizeof(base_salt), salt_seed);

    uint8_t *init_code = nullptr;
    size_t init_code_len = 0;
    if (!hex_to_bytes_alloc(init_code_hex_str, &init_code, &init_code_len)) {
        fprintf(stderr, "Invalid INIT_CODE hex string.\n");
        return 1;
    }

    uint8_t init_code_hash[32];
    const uint8_t empty_init_code = 0;
    const uint8_t *init_code_ptr = init_code_len ? init_code : &empty_init_code;
    keccak256(init_code_ptr, init_code_len, init_code_hash);

    cuda_check(cudaMemcpyToSymbol(kDeployerDev, deployer, sizeof(deployer)),
               "cudaMemcpyToSymbol deployer");
    cuda_check(cudaMemcpyToSymbol(kInitCodeHashDev, init_code_hash, sizeof(init_code_hash)),
               "cudaMemcpyToSymbol init_code_hash");
    cuda_check(cudaMemcpyToSymbol(kBaseSaltDev, base_salt, sizeof(base_salt)),
               "cudaMemcpyToSymbol base_salt");

    const int batch_size = 1 << 20;
    const int threads_per_block = 256;
    const uint64_t progress_interval = 1000000;

    printf("=== CREATE2 Contract Address Miner ===\n");
    print_gpu_info();

    char deployer_hex[41];
    char salt_hex[65];
    char init_hash_hex[65];
    bytes_to_hex(deployer, sizeof(deployer), deployer_hex);
    bytes_to_hex(base_salt, sizeof(base_salt), salt_hex);
    bytes_to_hex(init_code_hash, sizeof(init_code_hash), init_hash_hex);

    printf("Deployer: 0x%s\n", deployer_hex);
    printf("Base SALT (random start): 0x%s\n", salt_hex);
    printf("Init code bytes: %zu\n", init_code_len);
    printf("Init code hash: 0x%s\n", init_hash_hex);
    printf("Target: %d leading zeroes (hex)\n", target_zeroes);
    printf("Reporting: %d+ leading zeroes\n", report_zeroes);
    printf("Batch size: %d candidates\n", batch_size);

    uint64_t seed = get_random_seed();
    printf("Initial random offset: 0x%016llx\n-------------------------------------------------------\n",
           (unsigned long long)seed);
    const double expected_trials = std::pow(16.0, (double)target_zeroes);

    uint8_t *d_zeroes = nullptr;
    uint64_t *d_offsets = nullptr;
    cuda_check(cudaMalloc(&d_zeroes, batch_size * sizeof(uint8_t)), "cudaMalloc d_zeroes");
    cuda_check(cudaMalloc(&d_offsets, batch_size * sizeof(uint64_t)), "cudaMalloc d_offsets");

    uint8_t *h_zeroes = new uint8_t[batch_size];
    uint64_t *h_offsets = new uint64_t[batch_size];

    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    int best_zeroes = -1;
    uint64_t best_offset = 0;
    uint64_t total_checked = 0;
    uint64_t last_report = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    bool found = false;

    while (!found) {
        search_candidates<<<num_blocks, threads_per_block>>>(
            d_zeroes, d_offsets, seed, batch_size, total_checked);
        cuda_check(cudaGetLastError(), "search_candidates launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cuda_check(cudaMemcpy(h_zeroes, d_zeroes, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_zeroes");
        cuda_check(cudaMemcpy(h_offsets, d_offsets, batch_size * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_offsets");

        int found_index = -1;
        for (int i = 0; i < batch_size; i++) {
            int zeroes = h_zeroes[i];
            if (zeroes > best_zeroes) {
                best_zeroes = zeroes;
                best_offset = h_offsets[i];

                uint8_t candidate_salt[32];
                add_offset_to_salt_be(base_salt, best_offset, candidate_salt);

                uint8_t address[20];
                compute_create2_address(deployer, candidate_salt, init_code_hash, address);

                char salt_out[65];
                char address_hex[41];
                bytes_to_hex(candidate_salt, sizeof(candidate_salt), salt_out);
                bytes_to_hex(address, sizeof(address), address_hex);

                if (best_zeroes >= report_zeroes) {
                    printf("\nNew best: %d leading zeroes\n", best_zeroes);
                    printf("SALT: 0x%s\n", salt_out);
                    printf("Address: 0x%s\n", address_hex);
                    printf("-------------------------------------------------------\n");
                }

                if (best_zeroes >= target_zeroes) {
                    found = true;
                    found_index = i;
                    break;
                }
            }
        }

        if (found) {
            total_checked += (uint64_t)(found_index + 1);
            break;
        }

        total_checked += batch_size;

        if (total_checked - last_report >= progress_interval) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
            long secs = elapsed.count();
            if (secs == 0) secs = 1;
            double rate = total_checked / (double)secs;
            double mean_eta = expected_trials / rate;
            double no_hit = std::exp(-(double)total_checked / expected_trials);
            printf("Checked: %lu candidates | Speed: %.2f cand/s | Time: %lds | Mean ETA: %.0fs | No-hit: %.4f%%\r",
                   total_checked, rate, secs, mean_eta, no_hit * 100.0);
            fflush(stdout);
            last_report = total_checked;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    long secs = duration.count();
    if (secs == 0) secs = 1;

    uint8_t final_salt[32];
    add_offset_to_salt_be(base_salt, best_offset, final_salt);

    uint8_t final_address[20];
    compute_create2_address(deployer, final_salt, init_code_hash, final_address);

    char final_salt_hex[65];
    char final_address_hex[41];
    bytes_to_hex(final_salt, sizeof(final_salt), final_salt_hex);
    bytes_to_hex(final_address, sizeof(final_address), final_address_hex);

    printf("\n\nTARGET REACHED\n");
    printf("SALT: 0x%s\n", final_salt_hex);
    printf("Address: 0x%s\n", final_address_hex);
    printf("Leading zeroes: %d\n", best_zeroes);
    printf("Total checked: %lu candidates\n", total_checked);
    printf("Time elapsed: %ld seconds\n", secs);
    printf("Average speed: %.2f cand/s\n", total_checked / (double)secs);

    delete[] h_zeroes;
    delete[] h_offsets;
    delete[] init_code;
    cudaFree(d_zeroes);
    cudaFree(d_offsets);

    return 0;
}

// Compile & Run (tested on RTX 4070):
//   nvcc -arch=sm_89 -O3 --use_fast_math -Xcompiler -O3 -std=c++11 -diag-suppress=177 solidity_create2_miner.cu -o create2-miner && ./create2-miner

// result can be verified with e.g.:
//      cast create2 --deployer 0x4e59b44847b379578588920cA78FbF26c0B4956C --salt 0x<result> --init-code 0x<your-init-code>
