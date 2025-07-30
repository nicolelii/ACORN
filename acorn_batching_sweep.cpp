#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <set>
#include <algorithm>
#include <limits>
#include <cassert>
#include <chrono>

#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>

// Timing
// double elapsed() {
//     struct timeval tv;
//     gettimeofday(&tv, nullptr);
//     return tv.tv_sec + tv.tv_usec * 1e-6;
// }

// Read fvecs
float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) { perror(fname); exit(1); }
    int d; fread(&d, 1, sizeof(int), f);
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    size_t n = sz / ((d + 1) * 4);
    *d_out = d; *n_out = n;
    float* x = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        int dim; fread(&dim, sizeof(int), 1, f);
        fread(x + i * d, sizeof(float), d, f);
    }
    fclose(f); return x;
}

// Random metadata
std::vector<int> generate_random_metadata(size_t n, int gamma) {
    std::mt19937 gen(0);
    std::uniform_int_distribution<> dis(1, gamma);
    std::vector<int> metadata(n);
    for (size_t i = 0; i < n; i++) metadata[i] = dis(gen);
    return metadata;
}

std::vector<int> generate_query_attrs(size_t nq, int gamma) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, gamma);
    std::vector<int> attrs(nq);
    for (size_t i = 0; i < nq; i++) attrs[i] = dis(gen);
    return attrs;
}

std::vector<faiss::idx_t> compute_filtered_gt(const float* xq, size_t nq, size_t d,
    const float* xb, size_t nb,
    const std::vector<int>& metadata,
    const std::vector<int>& query_attrs,
    int k) {

    std::vector<faiss::idx_t> gt(nq * k);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, faiss::idx_t>> cands;
        for (size_t i = 0; i < nb; i++) {
            if (metadata[i] == query_attrs[q]) {
                float dist = 0;
                for (size_t j = 0; j < d; j++) {
                    float diff = xq[q * d + j] - xb[i * d + j];
                    dist += diff * diff;
                }
                cands.emplace_back(dist, i);
            }
        }
        std::sort(cands.begin(), cands.end());
        for (int i = 0; i < k; i++) gt[q * k + i] = (i < cands.size() ? cands[i].second : -1);
    }
    return gt;
}

int main() {
    const size_t d = 128, N = 1000000;
    const int M = 32, efc = 40, gamma = 12, M_beta = 64, k = 10;
    const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};
    const std::vector<int> efSearch_vals = {1, 2, 4, 8, 16, 32, 64, 128};

    printf("====================\nACORN-gamma SIFT1M Sweep\n====================\n");
    printf("Parameters: d=%zu, M=%d, efc=%d, gamma=%d, M_beta=%d, k=%d\n",
           d, M, efc, gamma, M_beta, k);

    // double t0 = elapsed();
    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> metadata = generate_random_metadata(N, gamma);
    // printf("[%.3f s] Loading SIFT1M base vectors\n", elapsed() - t0);
    printf("[%.3f s] Loading SIFT1M base vectors\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    size_t nb, nq, d2;
    float* xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);
    assert(nb >= N && d2 == d);

    printf("Loaded %zu base vectors\n", N);
    // printf("[%.3f s] Loading SIFT1M query vectors\n", elapsed() - t0);
    printf("[%.3f s] Loading SIFT1M query vectors\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    float* xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    assert(d2 == d);
    printf("Loaded %zu query vectors\n", nq);

    std::vector<int> query_attrs = generate_query_attrs(nq, gamma);

    // printf("[%.3f s] Creating ACORN-gamma index\n", elapsed() - t0);
    printf("[%.3f s] Creating ACORN-gamma index\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    faiss::IndexACORNFlat acorn(d, M, gamma, metadata, M_beta);
    acorn.add(N, xb);

    //save to disk
    faiss::write_index(&acorn, "acorn.index");

    // printf("[%.3f s] Computing filtered ground truth\n", elapsed() - t0);
    printf("[%.3f s] Computing filtered ground truth\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    
    std::vector<faiss::idx_t> filtered_gt = compute_filtered_gt(xq, nq, d, xb, N, metadata, query_attrs, k);

    std::vector<char> filter_map(nq * N);
    for (size_t q = 0; q < nq; q++) {
        for (size_t i = 0; i < N; i++) {
            filter_map[q * N + i] = (metadata[i] == query_attrs[q]);
        }
    }

    std::vector<faiss::idx_t> results(k * nq);
    std::vector<float> distances(k * nq);

    printf("\n====================\nBATCH SIZE + efSearch SWEEP\n====================\n");
    printf("BatchSize\tefSearch\tTime(s)\t\tR@10\t\tQPS\n");
    printf("---------------------------------------------------------------\n");

    for (int batch_size : batch_sizes) {
        for (int ef : efSearch_vals) {
            faiss::SearchParametersACORN params;
            params.efSearch = ef;

            // double t1 = elapsed();
            auto t1 = std::chrono::steady_clock::now();
            for (size_t start = 0; start < nq; start += batch_size) {
                size_t cur = std::min((size_t)batch_size, nq - start);
                acorn.search(cur,
                             xq + start * d,
                             k,
                             distances.data() + start * k,
                             results.data() + start * k,
                             filter_map.data() + start * N,
                             &params);
            }
            // double t2 = elapsed();
            auto t2 = std::chrono::steady_clock::now();

            int n_10 = 0;
            for (size_t i = 0; i < nq; i++) {
                std::set<faiss::idx_t> gt_set;
                for (int j = 0; j < k; j++) if (filtered_gt[i * k + j] != -1) gt_set.insert(filtered_gt[i * k + j]);
                for (int j = 0; j < k; j++) if (gt_set.count(results[i * k + j])) { n_10++; break; }
            }

            float recall_10 = n_10 / float(nq);
            double search_time = std::chrono::duration<double>(t2 - t1).count();

            double qps = nq / search_time;
            printf("%-10d\t%-8d\t%-10.4f\t%-8.4f\t%-8.2f\n", batch_size, ef, search_time, recall_10, qps);

            // Print sample results for every efSearch setting
            // printf("\n====================\nSAMPLE RESULTS (batch_size=%d, efSearch=%d)\n====================\n", batch_size, ef);
            // for (int i = 0; i < 3 && i < nq; i++) {
            //     printf("Query %d (attr=%d):\n  ACORN-γ: ", i, query_attrs[i]);
            //     for (int j = 0; j < 5; j++) {
            //         faiss::idx_t res = results[i * k + j];
            //         if (res != -1)
            //             printf("%ld(%d) ", res, metadata[res]);
            //         else
            //             printf("-1(?) ");
            //     }
            //     printf("\n  GT: ");
            //     for (int j = 0; j < 5; j++) {
            //         faiss::idx_t gt = filtered_gt[i * k + j];
            //         if (gt != -1)
            //             printf("%ld(%d) ", gt, metadata[gt]);
            //         else
            //             printf("-1(?) ");
            //     }
            //     printf("\n\n");
            // }

        }
    }

    printf("\n[%.3f s] Experiment completed!\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    delete[] xb;
    delete[] xq;
    return 0;
}
