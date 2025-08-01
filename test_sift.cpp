#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <sys/time.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

#include <iostream>
#include <vector>
#include <cassert>
#include <string>
#include <algorithm>
#include <queue>
#include <set>

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
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

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

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

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <number_vecs> <gamma> <M> <M_beta>\n", argv[0]);
        exit(1);
    }

    size_t N = strtoul(argv[1], NULL, 10);
    int gamma = atoi(argv[2]);
    int M = atoi(argv[3]);
    int M_beta = atoi(argv[4]);

    int k = 10;
    double t0 = elapsed();
    srand(0);

    // Define batch sizes and efSearch values to test
    const std::vector<int> batch_sizes = {32, 64, 128};
    const std::vector<int> efSearch_vals = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};

    printf("========== ACORN-GAMMA BATCH PROCESSING on SIFT1M ==========\n");
    printf("N = %ld, gamma = %d, M = %d, M_beta = %d\n", N, gamma, M, M_beta);

    std::vector<int> metadata = generate_random_metadata(N, gamma);
    printf("[%.3f s] Generated random metadata (%ld attrs)\n", elapsed() - t0, metadata.size());

    float* xq; size_t nq, d2;
    xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    size_t d = d2;
    printf("[%.3f s] Loaded %ld query vectors (dim = %ld)\n", elapsed() - t0, nq, d);

    std::vector<int> aq = generate_query_attrs(nq, gamma);
    printf("[%.3f s] Generated %ld query attributes\n", elapsed() - t0, aq.size());

    float* xb; size_t nb, d_check;
    xb = fvecs_read("sift/sift_base.fvecs", &d_check, &nb);
    assert(d == d_check);
    if (N < nb) nb = N;
    printf("[%.3f s] Loaded %ld base vectors (dim = %ld)\n", elapsed() - t0, nb, d);

    faiss::IndexACORNFlat acorn_index(d, M, gamma, metadata, M_beta);
    acorn_index.add(N, xb);
    printf("[%.3f s] Built ACORN-gamma index and added %ld vectors\n", elapsed() - t0, N);

    // Create filter map for all queries
    std::vector<char> filter_ids_map(nq * N);
    for (size_t i = 0; i < nq; ++i) {
        for (size_t j = 0; j < N; ++j) {
            filter_ids_map[i * N + j] = (metadata[j] == aq[i]);
        }
    }

    // DEBUG: Check bitmap selectivity for first few queries
    for (size_t q = 0; q < std::min(nq, size_t(5)); ++q) {
        int count = 0;
        for (size_t i = 0; i < N; i++) {
            if (filter_ids_map[q * N + i]) count++;
        }
        printf("[DEBUG] Query %zu: %d / %zu valid base vectors (%.4f%%)\n",
            q, count, N, 100.0 * count / N);
    }

    // Precompute ground-truth neighbors with filtering
    printf("[%.3f s] Computing ground truth...\n", elapsed() - t0);
    std::vector<faiss::idx_t> gt_nns(k * nq);
    std::vector<float> gt_dis(k * nq);

    for (size_t i = 0; i < nq; ++i) {
        int query_attr = aq[i];
        std::priority_queue<std::pair<float, faiss::idx_t>> heap;

        for (size_t j = 0; j < N; ++j) {
            if (metadata[j] != query_attr) continue;
            float dist = faiss::fvec_L2sqr(xq + i * d, xb + j * d, d);
            if (heap.size() < (size_t)k) {
                heap.emplace(dist, j);
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.emplace(dist, j);
            }
        }

        for (int j = k - 1; j >= 0; --j) {
            if (!heap.empty()) {
                gt_dis[i * k + j] = heap.top().first;
                gt_nns[i * k + j] = heap.top().second;
                heap.pop();
            } else {
                gt_dis[i * k + j] = HUGE_VALF;
                gt_nns[i * k + j] = -1;
            }
        }
    }
    printf("[%.3f s] Ground truth computed\n", elapsed() - t0);

    // Batch processing sweep
    std::vector<faiss::idx_t> results(k * nq);
    std::vector<float> distances(k * nq);

    printf("\n====================\nBATCH SIZE + efSearch SWEEP\n====================\n");
    printf("BatchSize\tefSearch\tTime(s)\t\tR@10\t\tQPS\n");
    printf("---------------------------------------------------------------\n");

    for (int batch_size : batch_sizes) {
        for (int efs : efSearch_vals) {
            acorn_index.acorn.efSearch = efs;

            double t1 = elapsed();
            
            // Process queries in batches
            for (size_t start = 0; start < nq; start += batch_size) {
                size_t cur_batch = std::min((size_t)batch_size, nq - start);
                
                acorn_index.search(cur_batch,
                                 xq + start * d,
                                 k,
                                 distances.data() + start * k,
                                 results.data() + start * k,
                                 filter_ids_map.data() + start * N);
            }
            
            double t2 = elapsed();

            // Compute Recall@10 against ground truth
            int total_correct = 0;
            int total_gt_items = 0;

            for (size_t i = 0; i < nq; ++i) {
                std::set<faiss::idx_t> gt_set;
                int valid_gt = 0;
                for (int j = 0; j < k; ++j) {
                    faiss::idx_t gt = gt_nns[i * k + j];
                    if (gt != -1) {
                        gt_set.insert(gt);
                        valid_gt++;
                    }
                }
                total_gt_items += valid_gt;

                for (int j = 0; j < k; ++j) {
                    if (gt_set.count(results[i * k + j])) {
                        total_correct++;
                    }
                }
            }

            float recall_10 = (total_gt_items > 0) ? float(total_correct) / total_gt_items : 0.0f;
            double search_time = t2 - t1;
            double qps = nq / search_time;

            printf("%-10d\t%-8d\t%-10.4f\t%-8.4f\t%-8.2f\n",
                   batch_size, efs, search_time, recall_10, qps);
        }
        printf("---------------------------------------------------------------\n");
    }

    // Debug output for first few queries to verify filtering
    if (false) { // Set to true if want to enable debug output
        printf("\n===== DEBUG: Sample Query Results =====\n");
        for (size_t q = 0; q < std::min(nq, size_t(3)); ++q) {
            printf("Query %zu with attr = %d:\n", q, aq[q]);
            for (int j = 0; j < k; ++j) {
                faiss::idx_t idx = results[q * k + j];
                if (idx < 0 || idx >= (faiss::idx_t)N) continue;
                int base_attr = metadata[idx];
                printf("  rank %d: id = %ld, attr = %d%s\n",
                    j, idx, base_attr,
                    base_attr == aq[q] ? " ✅" : " ❌");
            }
            printf("\n");
        }
    }

    delete[] xb;
    delete[] xq;

    printf("\n[%.3f s] ----- DONE -----\n", elapsed() - t0);
    return 0;
}