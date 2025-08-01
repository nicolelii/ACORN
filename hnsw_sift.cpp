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
#include <cfloat>

#include <faiss/index_io.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/utils/distances.h>

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
    std::vector<faiss::idx_t> gt(nq * k, -1);
    for (size_t q = 0; q < nq; q++) {
        std::vector<std::pair<float, faiss::idx_t>> cands;
        for (size_t i = 0; i < nb; i++) {
            if (metadata[i] == query_attrs[q]) {
                float dist = faiss::fvec_L2sqr(xq + q * d, xb + i * d, d);
                cands.emplace_back(dist, i);
            }
        }
        std::partial_sort(cands.begin(), cands.begin() + std::min((size_t)k, cands.size()), cands.end());
        for (int i = 0; i < k && i < cands.size(); i++) {
            gt[q * k + i] = cands[i].second;
        }
    }
    return gt;
}

int main() {
    const size_t d = 128, N = 1000000;
    const int M = 32, efc = 40, gamma = 12, k = 10;
    const std::vector<int> efSearch_vals = {1,2,10,20,30,50,100,800};
    const std::vector<int> batch_sizes = {32, 64, 128};
    auto t0 = std::chrono::steady_clock::now();

    // Load data
    size_t nb, nq, d2;

    printf("[%.3f s] Loading base vectors\n",std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    float* xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);

    printf("[%.3f s] Loading query vectors\n",std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    float* xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    assert(nb >= N && d2 == d);

    std::vector<int> metadata = generate_random_metadata(N, gamma);
    std::vector<int> query_attrs = generate_query_attrs(nq, gamma);

    printf("[%.3f s] Building HNSWs index\n",std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());


    faiss::IndexHNSWFlat* hnsw;
    try {
        faiss::Index* base = faiss::read_index("hnsw_sift_experiment.index");
        hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(base);
        if (!hnsw) {
            throw std::runtime_error("Index is not an IndexHNSWFlat");
        }
    } catch (...) {
        hnsw = new faiss::IndexHNSWFlat(d, M);
        hnsw->hnsw.efConstruction = efc;
        hnsw->add(N, xb);
        faiss::write_index(hnsw, "hnsw_sift_experiment.index");
    }

    std::vector<faiss::idx_t> gt = compute_filtered_gt(xq, nq, d, xb, N, metadata, query_attrs, k);

    for (int bs : batch_sizes) {
        printf("=== Batch Size: %d ===\n", bs);
        printf("efSearch\tRecall@10\tQPS\n");

        for (int ef : efSearch_vals) {
            hnsw->hnsw.efSearch = ef;
            auto start = std::chrono::steady_clock::now();

            int total_correct = 0;
            int total_relevant = 0;
            std::vector<faiss::idx_t> results(nq * k);

            for (size_t start_idx = 0; start_idx < nq; start_idx += bs) {
                size_t cur_bs = std::min((size_t)bs, nq - start_idx);
                int over_k = 120;

                std::vector<float> D(cur_bs * over_k);
                std::vector<faiss::idx_t> I(cur_bs * over_k);

                hnsw->search(cur_bs, xq + start_idx * d, over_k, D.data(), I.data());

                for (size_t b = 0; b < cur_bs; b++) {
                    int attr = query_attrs[start_idx + b];
                    std::vector<std::pair<float, faiss::idx_t>> filtered;

                    for (int j = 0; j < over_k; j++) {
                        int idx = b * over_k + j;
                        if (I[idx] >= 0 && metadata[I[idx]] == attr) {
                            filtered.emplace_back(D[idx], I[idx]);
                        }
                    }

                    std::sort(filtered.begin(), filtered.end());
                    for (int j = 0; j < k; j++) {
                        results[(start_idx + b) * k + j] = (j < filtered.size()) ? filtered[j].second : -1;
                    }

                    // Calculate recall@10 for this query
                    std::set<faiss::idx_t> gt_set;
                    int relevant_count = 0;
                    for (int j = 0; j < k; j++) {
                        if (gt[(start_idx + b) * k + j] != -1) {
                            gt_set.insert(gt[(start_idx + b) * k + j]);
                            relevant_count++;
                        }
                    }
                    
                    if (relevant_count > 0) {
                        int correct_for_query = 0;
                        for (int j = 0; j < k; j++) {
                            if (results[(start_idx + b) * k + j] != -1 && 
                                gt_set.count(results[(start_idx + b) * k + j])) {
                                correct_for_query++;
                            }
                        }
                        total_correct += correct_for_query;
                        total_relevant += relevant_count;
                    }
                }
            }

            auto end = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(end - start).count();
            double recall = (total_relevant > 0) ? double(total_correct) / double(total_relevant) : 0.0;
            double qps = nq / elapsed;
            printf("%d\t\t%.4f\t\t%.2f\n", ef, recall, qps);
        }
    }

    delete[] xb;
    delete[] xq;
    delete hnsw;
    return 0;
}