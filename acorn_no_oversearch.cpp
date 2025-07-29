#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cassert>
#include <cstring>
#include <unordered_set>
#include <cstddef>

#include "faiss/IndexACORN.h"
#include "faiss/impl/ACORN.h"

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) return nullptr;
    int d;
    fread(&d, 1, sizeof(int), f);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t n = sz / ((d + 1) * 4);
    float* data = new float[n * (d + 1)];
    fread(data, sizeof(float), n * (d + 1), f);
    fclose(f);
    float* x = new float[n * d];
    for (size_t i = 0; i < n; i++) {
        memcpy(x + i * d, data + 1 + i * (d + 1), d * sizeof(float));
    }
    delete[] data;
    *d_out = d;
    *n_out = n;
    return x;
}

int* ivecs_read(const char* fname, size_t* nq_out, size_t* k_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) return nullptr;
    int k;
    fread(&k, 1, sizeof(int), f);
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t nq = sz / ((k + 1) * 4);
    int* all = new int[nq * (k + 1)];
    fread(all, sizeof(int), nq * (k + 1), f);
    fclose(f);
    int* gt = new int[nq * k];
    for (size_t i = 0; i < nq; i++) {
        memcpy(gt + i * k, all + i * (k + 1) + 1, k * sizeof(int));
    }
    delete[] all;
    *nq_out = nq;
    *k_out = k;
    return gt;
}

int main() {
    int d = 128;
    int M = 32;
    int efc = 40;
    int M_beta = 64;
    int gamma = 12;
    int k = 10;
    int batch_size = 10000;

    float sift1m_selectivity = 1.0f / 12.0f;

    size_t nb, nq, d2;

    float* xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);
    assert(d == d2);

    float* xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    assert(d == d2);

    size_t gt_nq, gt_k;
    int* gt = ivecs_read("sift/sift_groundtruth.ivecs", &gt_nq, &gt_k);
    assert(gt_nq == nq);

    std::cout << "=== ACORN-γ Benchmark ===" << std::endl;
    std::cout << "Parameters: M=" << M << ", efc=" << efc << ", M_β=" << M_beta << ", γ=" << gamma << std::endl;

    // simulate structured data for 12 classes
    std::vector<int> metadata(nb);
    int num_classes = 12;
    size_t block_size = nb / num_classes;
    for (size_t i = 0; i < nb; ++i) {
        metadata[i] = std::min((int)(i / block_size), num_classes - 1);
    }

    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    index.acorn.efConstruction = efc;

    // build acorn index
    auto build_start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb);
    auto build_end = std::chrono::high_resolution_clock::now();
    std::cout << "Index built in " << std::chrono::duration<double>(build_end - build_start).count() << " seconds" << std::endl;

    // define structured predicate, only allow class 0
    // for each (qi, bi) pair, it checks if bi's class label (from metadata[bi]) is in the allowed set.
    // if yes, mark filter[qi * nb + bi] = 1 to indicate that vector bi is a valid candidate for query qi.
    std::unordered_set<int> allowed_classes = {0};
    std::vector<char> filter(nq * nb, 0);
    for (size_t qi = 0; qi < nq; ++qi) {
        for (size_t bi = 0; bi < nb; ++bi) {
            if (allowed_classes.count(metadata[bi])) {
                filter[qi * nb + bi] = 1;
            }
        }
    }

    size_t total_allowed = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        if (allowed_classes.count(metadata[bi])) total_allowed++;
    }
    float actual_selectivity = (float)total_allowed / nb;

    std::cout << "Selectivity check: " << (actual_selectivity * 100) << "% (" << total_allowed << "/" << nb << ")" << std::endl;

    std::cout << "\nefSearch,QPS,Recall@10,us/query,Avg_Batch_Time_ms\n";

    std::vector<faiss::idx_t> nns(batch_size * k, -1); // result buffer for nearest neighbor indices
    std::vector<float> dis(batch_size * k, 0.0f); // buffer for distances
    std::vector<int> ef_values;
    for (int ef = 1; ef <= 2048; ef *= 2) ef_values.push_back(ef);

    for (int ef : ef_values) {
        faiss::SearchParametersACORN params;
        params.efSearch = ef;

        double total_search_time = 0.0;
        int total_correct = 0, total_evaluated = 0;
        std::vector<double> batch_times;

        int num_batches = (nq + batch_size - 1) / batch_size;
        for (int b = 0; b < num_batches; ++b) {
            int start = b * batch_size;
            int end = std::min(start + batch_size, (int)nq);
            int curr_size = end - start;

            std::fill(nns.begin(), nns.begin() + curr_size * k, -1);
            std::fill(dis.begin(), dis.begin() + curr_size * k, 0.0f);
            float* xq_batch = xq + start * d; // pointer to the current batch of queries
            char* f_batch = filter.data() + start * nb; // pointer to the corresponding portion of the filter mask

            auto t0 = std::chrono::high_resolution_clock::now();
            index.search(curr_size, xq_batch, k, dis.data(), nns.data(), f_batch, &params);
            auto t1 = std::chrono::high_resolution_clock::now();

            double t = std::chrono::duration<double>(t1 - t0).count();
            total_search_time += t;
            batch_times.push_back(t * 1000.0);

            for (int i = 0; i < curr_size; ++i) {
                int gidx = start + i;
                std::vector<int> valid_gt;
                for (int j = 0; j < std::min(k, (int)gt_k); ++j) {
                    int id = gt[gidx * gt_k + j];
                    if (id < nb && allowed_classes.count(metadata[id])) valid_gt.push_back(id);
                } // extract the top-k ground truth neighbors for this query, filtered to only include those allowed by the predicate 

                if (valid_gt.empty()) continue;
                total_evaluated++;

                // check if any of the predicted top-k neighbors match the valid ground truth set.
                for (int j = 0; j < k; ++j) {
                    if (std::find(valid_gt.begin(), valid_gt.end(), nns[i * k + j]) != valid_gt.end()) {
                        total_correct++;
                        break;
                    }
                }
            }
        }

        double recall = total_evaluated ? (double)total_correct / total_evaluated : 0.0;
        double qps = nq / total_search_time;
        double us_per_query = (total_search_time * 1e6) / nq;
        double avg_batch_ms = 0;
        for (auto& t : batch_times) avg_batch_ms += t;
        avg_batch_ms /= batch_times.size();

        std::cout << ef << "," << qps << "," << recall << "," << us_per_query << "," << avg_batch_ms << "\n";
    }

    delete[] xb;
    delete[] xq;
    delete[] gt;
    return 0;
}
