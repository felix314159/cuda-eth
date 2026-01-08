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

// set your function name + parameters that you need to mine a number suffix for
#define PREFIX_STR "batchWithdraw_"
#define SUFFIX_STR "(uint256)"
__device__ __constant__ char kPrefixDev[] = PREFIX_STR;
__device__ __constant__ char kSuffixDev[] = SUFFIX_STR;

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

HD int count_leading_zero_nibbles(const uint8_t *hash) {
    int count = 0;
    for (int i = 0; i < 32; i++) {
        uint8_t b = hash[i];
        if (b == 0) {
            count += 2;
            continue;
        }
        if ((b >> 4) == 0) count += 1;
        break;
    }
    return count;
}

// -------------------- generate candidates (unique decimal enumeration) ----------------------
static const char kPrefixHost[] = PREFIX_STR;
static const char kSuffixHost[] = SUFFIX_STR;
static constexpr int kPrefixLen = sizeof(PREFIX_STR) - 1;
static constexpr int kSuffixLen = sizeof(SUFFIX_STR) - 1;
static constexpr int kMaxDigits = 15; // find 15 or less digits that leads to all zeroes selector
static constexpr int kMaxMsgLen = kPrefixLen + kMaxDigits + kSuffixLen;

__global__ void search_candidates(uint8_t *out_zeroes, uint64_t *out_digits,
                                  uint64_t seed, int num_candidates, uint64_t nonce, uint64_t total_space) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    uint64_t start = seed % total_space;
    uint64_t candidate_id = start + nonce + (uint64_t)idx;
    if (candidate_id >= total_space) {
        candidate_id %= total_space; // wraps after ~1.1e15 attempts
    }

    uint64_t remaining = candidate_id;
    uint64_t range = 10;
    int len = 1;
    while (len < kMaxDigits && remaining >= range) {
        remaining -= range;
        len++;
        range *= 10;
    }

    uint8_t digits[kMaxDigits];
    for (int i = len - 1; i >= 0; i--) {
        digits[i] = (uint8_t)(remaining % 10);
        remaining /= 10;
    }

    uint64_t packed = ((uint64_t)len) << 60;
    char msg[kMaxMsgLen];
    int pos = 0;

    for (int i = 0; i < kPrefixLen; i++) msg[pos++] = kPrefixDev[i];

    for (int i = 0; i < len; i++) {
        uint8_t digit = digits[i];
        msg[pos++] = (char)('0' + digit);
        packed |= ((uint64_t)digit & 0xFULL) << (i * 4);
    }

    for (int i = 0; i < kSuffixLen; i++) msg[pos++] = kSuffixDev[i];

    uint8_t hash[32];
    keccak256((const uint8_t*)msg, (size_t)pos, hash);

    out_zeroes[idx] = (uint8_t)count_leading_zero_nibbles(hash);
    out_digits[idx] = packed;
}

static int build_candidate_string(uint64_t packed, char *out, size_t out_size) {
    const int needed = kPrefixLen + kSuffixLen + kMaxDigits + 1;
    if ((int)out_size < needed) return -1;

    int len = (int)((packed >> 60) & 0xF);
    if (len < 1 || len > kMaxDigits) return -1;

    char digits[kMaxDigits];
    for (int i = 0; i < len; i++) {
        uint8_t digit = (uint8_t)((packed >> (i * 4)) & 0xF);
        if (digit > 9) return -1;
        digits[i] = (char)('0' + digit);
    }

    int pos = 0;
    memcpy(out + pos, kPrefixHost, kPrefixLen);
    pos += kPrefixLen;
    memcpy(out + pos, digits, len);
    pos += len;
    memcpy(out + pos, kSuffixHost, kSuffixLen);
    pos += kSuffixLen;
    out[pos] = '\0';
    return pos;
}

static void hash_to_hex(const uint8_t *hash, char *hex_str) {
    for (int i = 0; i < 32; i++) {
        sprintf(hex_str + i * 2, "%02x", hash[i]);
    }
    hex_str[64] = '\0';
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

static uint64_t compute_total_space() {
    uint64_t total = 0;
    uint64_t pow10 = 1;
    for (int i = 0; i < kMaxDigits; i++) {
        pow10 *= 10;
        total += pow10;
    }
    return total;
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
    // set your custom fn sig in lines 18+19
    const int target_zeroes = 8;
    const int report_zeroes = 6;
    const int batch_size = 1 << 20;
    const int threads_per_block = 256;
    const uint64_t progress_interval = 1000000;

    printf("=== Keccak256 Function Signature Miner ===\n");
    print_gpu_info();
    printf("Template: %sNUMBERS%s\n", kPrefixHost, kSuffixHost);
    printf("Digits: up to %d\n", kMaxDigits);
    printf("Target: %d leading zeroes (hex)\n", target_zeroes);
    printf("Reporting: %d+ leading zeroes\n", report_zeroes);
    printf("Batch size: %d candidates\n", batch_size);

    uint64_t seed = get_random_seed();
    printf("Initial random seed: 0x%016llx\n-------------------------------------------------------\n-------------------------------------------------------\n",
           (unsigned long long)seed);
    const uint64_t total_space = compute_total_space();
    const double expected_trials = std::pow(16.0, (double)target_zeroes);

    uint8_t *d_zeroes = nullptr;
    uint64_t *d_digits = nullptr;
    cuda_check(cudaMalloc(&d_zeroes, batch_size * sizeof(uint8_t)), "cudaMalloc d_zeroes");
    cuda_check(cudaMalloc(&d_digits, batch_size * sizeof(uint64_t)), "cudaMalloc d_digits");

    uint8_t *h_zeroes = new uint8_t[batch_size];
    uint64_t *h_digits = new uint64_t[batch_size];

    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;

    int best_zeroes = -1;
    uint64_t best_packed = 0;
    uint64_t total_checked = 0;
    uint64_t last_report = 0;

    auto start_time = std::chrono::high_resolution_clock::now();
    bool found = false;

    while (!found) {
        search_candidates<<<num_blocks, threads_per_block>>>(
            d_zeroes, d_digits, seed, batch_size, total_checked, total_space);
        cuda_check(cudaGetLastError(), "search_candidates launch");
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        cuda_check(cudaMemcpy(h_zeroes, d_zeroes, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_zeroes");
        cuda_check(cudaMemcpy(h_digits, d_digits, batch_size * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy h_digits");

        int found_index = -1;
        for (int i = 0; i < batch_size; i++) {
            int zeroes = h_zeroes[i];
            if (zeroes > best_zeroes) {
                best_zeroes = zeroes;
                best_packed = h_digits[i];

                char candidate[64];
                if (build_candidate_string(best_packed, candidate, sizeof(candidate)) < 0) {
                    continue;
                }

                uint8_t hash[32];
                keccak256((const uint8_t*)candidate, strlen(candidate), hash);

                char hash_hex[65];
                hash_to_hex(hash, hash_hex);

                if (best_zeroes >= report_zeroes) {
                    printf("\nNew best: %d leading zeroes with %s\n", best_zeroes, candidate);
                    printf("Hash: 0x%s\n", hash_hex);
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

    char final_candidate[64];
    build_candidate_string(best_packed, final_candidate, sizeof(final_candidate));

    uint8_t final_hash[32];
    keccak256((const uint8_t*)final_candidate, strlen(final_candidate), final_hash);

    char final_hash_hex[65];
    hash_to_hex(final_hash, final_hash_hex);

    printf("\n\nTARGET REACHED\n");
    printf("Best candidate: %s\n", final_candidate);
    printf("Hash: 0x%s\n", final_hash_hex);
    printf("Leading zeroes: %d\n", best_zeroes);
    printf("Total checked: %lu candidates\n", total_checked);
    printf("Time elapsed: %ld seconds\n", secs);
    printf("Average speed: %.2f cand/s\n", total_checked / (double)secs);

    delete[] h_zeroes;
    delete[] h_digits;
    cudaFree(d_zeroes);
    cudaFree(d_digits);

    return 0;
}

// Compile & Run (tested on RTX 4070):
//      nvcc -arch=sm_89 -O3 --use_fast_math -Xcompiler -O3 -std=c++11 -diag-suppress=177 solidity_function_name_miner.cu -o function-miner && ./function-miner

// finding 7 leading zeroes is instant (less than 2 sec), finding 8 leading zeroes (an all zeroes selector) takes around 120 sec on my machine
// e.g.
//  -> keccak256("batchWithdraw_90385706483102(uint256)")   = 00000000fecc622b385dbedb6e38b1649896f95c923efd588f1e1966e313dbf8
//  -> keccak256("batchWithdraw_99546325256525(uint256)")   = 00000000df8b8bb70257f6daf840cb3393094cfb96329b463916605e3792b2f5
//
// if u put both in a contract solc will tell u: `Error: Function signature hash collision for batchWithdraw_90385706483102(uint256)`:
/*

// SPDX-License-Identifier: GPL-3.0
pragma solidity ^0.8.33;

contract C {
    // zero
    function batchWithdraw_99546325256525(uint256) pure public {
        return;
    }

    // zero
    function batchWithdraw_90385706483102(uint256) pure public {
        revert();
    }
    
    // non-zero but u might think its zero if u dont know uint is actually hashed as uint256
    function batchWithdraw_08891485022903(uint) pure public {
        revert();
    }

    receive() external payable {}

    fallback() external payable {}
}

*/
