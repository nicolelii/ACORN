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
    // === Exact ACORN-γ Parameters from Paper ===
    int d = 128;
    int M = 32;        // Same as HNSW baseline (FAISS default)
    int efc = 40;      // Same as HNSW baseline (FAISS default)  
    int M_beta = 64;   // 2*M for SIFT1M (paper specifies this)
    int gamma = 12;    // Expected minimum selectivity for SIFT1M
    int k = 10;        // Target number of results
    
    // === CONFIGURABLE BATCH SIZE ===
    int batch_size = 10000;  
    //1 (single query), 10, 50, 100, 500, 1000, 10000 (all at once)
    
    // Precise selectivity: 1/12 = 8.333...%
    float sift1m_selectivity = 1.0f / 12.0f;  // Exactly 8.33%
    
    size_t nb, nq, d2;

    // === Load data ===
    float* xb = fvecs_read("sift/sift_base.fvecs", &d2, &nb);
    assert(d == d2);
    
    float* xq = fvecs_read("sift/sift_query.fvecs", &d2, &nq);
    assert(d == d2);
    
    size_t gt_nq, gt_k;
    int* gt = ivecs_read("sift/sift_groundtruth.ivecs", &gt_nq, &gt_k);
    assert(gt_nq == nq);

    std::cout << "=== ACORN-γ Implementation with Configurable Batch Processing ===" << std::endl;
    std::cout << "Dataset: SIFT1M (" << nb << " base, " << nq << " queries)" << std::endl;
    std::cout << "Batch size: " << batch_size << " queries per batch" << std::endl;
    std::cout << "Number of batches: " << ((nq + batch_size - 1) / batch_size) << std::endl;
    std::cout << "Parameters: M=" << M << ", efc=" << efc << ", M_β=" << M_beta << ", γ=" << gamma << std::endl;
    std::cout << "Selectivity: " << (sift1m_selectivity * 100) << "% (1/12 = exactly 8.333%)" << std::endl;

    // === Create metadata with 12 classes (to get exact 1/12 selectivity) ===
    std::vector<int> metadata(nb);
    int num_classes = 12;  
    size_t block_size = nb / num_classes;
    for (size_t i = 0; i < nb; ++i) {
        metadata[i] = std::min((int)(i / block_size), num_classes - 1);
    }

    // === Build ACORN-γ index with exact paper parameters ===
    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);
    index.acorn.efConstruction = efc;
    
    auto build_start = std::chrono::high_resolution_clock::now();
    index.add(nb, xb);
    auto build_end = std::chrono::high_resolution_clock::now();
    
    double build_time = std::chrono::duration<double>(build_end - build_start).count();
    std::cout << "Index built in " << build_time << " seconds" << std::endl;

    // === Setup filtering for exactly 1 out of 12 classes ===
    int num_allowed = 1;  // Select exactly 1 class out of 12
    std::unordered_set<int> allowed_classes;
    allowed_classes.insert(0);  // Allow only class 0

    // Pre-build filter matrix
    std::vector<char> filter(nq * nb, 0);
    for (size_t qi = 0; qi < nq; ++qi) {
        for (size_t bi = 0; bi < nb; ++bi) {
            if (allowed_classes.count(metadata[bi])) {
                filter[qi * nb + bi] = 1;
            }
        }
    }

    // Verify exact selectivity
    size_t total_allowed = 0;
    for (size_t bi = 0; bi < nb; ++bi) {
        if (allowed_classes.count(metadata[bi])) {
            total_allowed++;
        }
    }
    float actual_selectivity = (float)total_allowed / nb;
    
    std::cout << "Filter verification: " << total_allowed << "/" << nb << " vectors allowed" << std::endl;
    std::cout << "Actual selectivity: " << (actual_selectivity * 100) << "%" << std::endl;
    std::cout << "Target selectivity: " << (sift1m_selectivity * 100) << "%" << std::endl;

    // === Paper's exact methodology with batch processing ===
    std::cout << "\n=== Generating Recall-QPS Curve with Batch Processing ===" << std::endl;
    std::cout << "Following HNSW post-filtering: over-search by K/s = " << k << "/" << sift1m_selectivity << " = " << (k/sift1m_selectivity) << std::endl;
    
    // Paper's exact approach: over-search by K/s
    int K_oversearch = static_cast<int>(std::ceil(k / sift1m_selectivity));
    
    std::cout << "Using K_oversearch = " << K_oversearch << std::endl;
    std::cout << "Processing in batches of " << batch_size << " queries" << std::endl;
    std::cout << "\nResults:" << std::endl;
    std::cout << "efSearch,QPS,Recall@10,QPS_per_query_us,Avg_Batch_Time_ms" << std::endl;
    
    // Pre-allocate result vectors for batch processing
    std::vector<faiss::idx_t> nns(batch_size * K_oversearch, -1);
    std::vector<float> dis(batch_size * K_oversearch, 0.0f);
    
    std::vector<int> ef_values;
    for (int ef = 1; ef <= 2048; ef *= 2) {
        ef_values.push_back(ef);  // Powers of 2: 1, 2, 4, 8, ..., 2048
    }
    
    for (int ef : ef_values) {
        faiss::SearchParametersACORN params;
        params.efSearch = ef;
        
        // Track total time and results across all batches
        double total_search_time = 0.0;
        int total_correct = 0;
        int total_evaluated_queries = 0;
        std::vector<double> batch_times;
        
        // Calculate number of batches needed
        int num_batches = (nq + batch_size - 1) / batch_size;
        
        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Calculate batch boundaries
            int start_idx = batch_idx * batch_size;
            int end_idx = std::min(start_idx + batch_size, (int)nq);
            int current_batch_size = end_idx - start_idx;
            
            // Clear results from previous iteration
            std::fill(nns.begin(), nns.begin() + current_batch_size * K_oversearch, -1);
            std::fill(dis.begin(), dis.begin() + current_batch_size * K_oversearch, 0.0f);
            
            // Get pointers to current batch data
            float* batch_xq = xq + start_idx * d;
            char* batch_filter = filter.data() + start_idx * nb;
            
            // Measure search time for this batch
            auto batch_start = std::chrono::high_resolution_clock::now();
            index.search(current_batch_size, batch_xq, K_oversearch, dis.data(), nns.data(), batch_filter, &params);
            auto batch_end = std::chrono::high_resolution_clock::now();
            
            double batch_time = std::chrono::duration<double>(batch_end - batch_start).count();
            total_search_time += batch_time;
            batch_times.push_back(batch_time * 1000.0); // Convert to ms
            
            // Calculate recall for this batch
            for (int i = 0; i < current_batch_size; ++i) {
                int global_query_idx = start_idx + i;
                
                // Find ground truth that passes the filter
                std::vector<int> compatible_gt;
                for (int j = 0; j < std::min(k, (int)gt_k); ++j) {
                    int gt_id = gt[global_query_idx * gt_k + j];
                    if (gt_id < nb && allowed_classes.count(metadata[gt_id])) {
                        compatible_gt.push_back(gt_id);
                    }
                }
                
                if (compatible_gt.empty()) continue;
                total_evaluated_queries++;
                
                // Check if any compatible GT is in our top-k results
                bool found = false;
                for (int gt_id : compatible_gt) {
                    for (int l = 0; l < std::min(k, K_oversearch); ++l) {
                        if (nns[i * K_oversearch + l] == gt_id) {
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }
                
                if (found) total_correct++;
            }
        }
        
        // Calculate overall metrics
        double qps = nq / total_search_time;
        double us_per_query = (total_search_time * 1000000.0) / nq;
        double recall = total_evaluated_queries > 0 ? (double)total_correct / total_evaluated_queries : 0.0;
        
        // Calculate average batch time
        double avg_batch_time_ms = 0.0;
        for (double bt : batch_times) {
            avg_batch_time_ms += bt;
        }
        avg_batch_time_ms /= batch_times.size();
        
        std::cout << ef << "," << qps << "," << recall << "," << us_per_query << "," << avg_batch_time_ms << std::endl;
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "This implementation processes queries in batches of " << batch_size << ":" << std::endl;
    std::cout << "1. ACORN-γ with M=32, efc=40, M_β=64, γ=12" << std::endl;
    std::cout << "2. 12 classes total, 1 class allowed = 1/12 = 8.333%" << std::endl;
    std::cout << "3. Over-search by K/s = " << K_oversearch << " candidates" << std::endl;
    std::cout << "4. Process " << batch_size << " queries at a time" << std::endl;
    std::cout << "5. Total batches processed: " << ((nq + batch_size - 1) / batch_size) << std::endl;
    std::cout << "6. QPS calculated across all " << nq << " queries" << std::endl;

    // Cleanup
    delete[] xb;
    delete[] xq;
    delete[] gt;
    
    return 0;
}