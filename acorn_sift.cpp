#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <vector>
#include <iostream>
#include <set>
#include <algorithm>
#include <limits>
#include <cassert>
#include <chrono>
#include <queue>
#include <unordered_map>

#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexACORN.h>
#include <faiss/utils/distances.h>

using Attr = int;

struct Cand {
    float dist;
    faiss::idx_t id;
};
struct WorseFirst {
    bool operator()(const Cand& a, const Cand& b) const { return a.dist < b.dist; }
};

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

std::unordered_map<Attr, std::vector<faiss::idx_t>> build_attr_buckets(const std::vector<int>& metadata) {
    std::unordered_map<Attr, std::vector<faiss::idx_t>> buckets;
    for (faiss::idx_t i = 0; i < (faiss::idx_t)metadata.size(); ++i) {
        buckets[metadata[i]].push_back(i);
    }
    return buckets;
}

std::vector<faiss::idx_t> compute_filtered_gt(const float* xq, size_t nq, size_t d,
                                              const float* xb, size_t /*nb*/,
                                              const std::vector<int>& metadata,
                                              const std::vector<int>& query_attrs,
                                              int k) {
    auto buckets = build_attr_buckets(metadata);
    std::vector<faiss::idx_t> gt(nq * k, -1);
    if (k <= 0) return gt;

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t q = 0; q < (ptrdiff_t)nq; ++q) {
        const float* qvec = xq + (size_t)q * d;
        auto it = buckets.find(query_attrs[q]);
        if (it == buckets.end()) continue;

        std::priority_queue<Cand, std::vector<Cand>, WorseFirst> topk;
        for (faiss::idx_t idx : it->second) {
            const float* bvec = xb + (size_t)idx * d;
            float dist = faiss::fvec_L2sqr(qvec, bvec, (int)d);
            if ((int)topk.size() < k) {
                topk.push({dist, idx});
            } else if (dist < topk.top().dist) {
                topk.pop();
                topk.push({dist, idx});
            }
        }

        int out = std::min((int)topk.size(), k);
        for (int i = out - 1; i >= 0; --i) {
            gt[q * k + i] = topk.top().id;
            topk.pop();
        }
    }
    return gt;
}

int main() {
    const size_t d = 128, N = 1000000;
    const int M = 32, efc = 40, gamma = 12, M_beta = 64, k = 10;
    const std::vector<int> batch_sizes = {32, 64, 128};
    const std::vector<int> efSearch_vals = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};

    auto t0 = std::chrono::steady_clock::now();

    std::vector<int> metadata = generate_random_metadata(N, gamma);
    printf("[%.3f s] Loading SIFT1M base vectors\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    size_t nb, nq, d2;
    float* xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);
    assert(nb >= N && d2 == d);

    printf("[%.3f s] Loading SIFT1M query vectors\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    float* xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    assert(d2 == d);

    std::vector<int> query_attrs = generate_query_attrs(nq, gamma);

    printf("[%.3f s] Creating ACORN-gamma index\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    faiss::IndexACORNFlat* acorn;
    try {
        faiss::Index* base = faiss::read_index("acorn.index");
        acorn = dynamic_cast<faiss::IndexACORNFlat*>(base);
    } catch (...) {
        acorn = new faiss::IndexACORNFlat(d, M, gamma, metadata, M_beta);
        acorn->add(N, xb);
        faiss::write_index(acorn, "acorn.index");
    }

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

            auto t1 = std::chrono::steady_clock::now();
            for (size_t start = 0; start < nq; start += batch_size) {
                size_t cur = std::min((size_t)batch_size, nq - start);
                acorn->search(cur,
                              xq + start * d,
                              k,
                              distances.data() + start * k,
                              results.data() + start * k,
                              filter_map.data() + start * N,
                              &params);
            }
            auto t2 = std::chrono::steady_clock::now();

            int total_correct = 0;
            int total_gt_items = 0;

            for (size_t i = 0; i < nq; i++) {
                std::set<faiss::idx_t> gt_set;
                int valid_gt = 0;
                for (int j = 0; j < k; j++) {
                    faiss::idx_t gt = filtered_gt[i * k + j];
                    if (gt != -1) {
                        gt_set.insert(gt);
                        valid_gt++;
                    }
                }
                total_gt_items += valid_gt;

                for (int j = 0; j < k; j++) {
                    if (gt_set.count(results[i * k + j])) {
                        total_correct++;
                    }
                }
            }

            float recall_10 = (total_gt_items > 0) ? float(total_correct) / total_gt_items : 0.0f;
            double search_time = std::chrono::duration<double>(t2 - t1).count();
            double qps = nq / search_time;

            printf("%-10d\t%-8d\t%-10.4f\t%-8.4f\t%-8.2f\n",
                   batch_size, ef, search_time, recall_10, qps);
        }
    }

    printf("\n[%.3f s] Experiment completed!\n",
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    delete[] xb;
    delete[] xq;
    return 0;
}
