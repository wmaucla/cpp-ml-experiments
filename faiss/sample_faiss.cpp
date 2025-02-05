#include <faiss/IndexFlat.h>
#include <iostream>
#include <vector>

int main() {
    // Dimensions of vectors
    int d = 128;  // Vector dimensionality
    int nb = 10000; // Database size
    int nq = 5;   // Number of query vectors

    // Generate random database vectors
    std::vector<float> database(nb * d);
    for (int i = 0; i < nb * d; ++i) {
        database[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Generate random query vectors
    std::vector<float> queries(nq * d);
    for (int i = 0; i < nq * d; ++i) {
        queries[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Create a FAISS index (L2-based)
    faiss::IndexFlatL2 index(d); // Flat index with L2 (Euclidean distance)
    std::cout << "Is index trained? " << index.is_trained << std::endl;

    // Add database vectors to index
    index.add(nb, database.data());
    std::cout << "Total vectors in index: " << index.ntotal << std::endl;

    // Search for nearest neighbors
    int k = 5;  // Number of nearest neighbors
    std::vector<faiss::idx_t> indices(nq * k); // Nearest neighbor indices
    std::vector<float> distances(nq * k); // Distances to nearest neighbors

    index.search(nq, queries.data(), k, distances.data(), indices.data());

    // Print search results
    for (int i = 0; i < nq; ++i) {
        std::cout << "Query " << i << " nearest neighbors:\n";
        for (int j = 0; j < k; ++j) {
            std::cout << "  Rank " << j + 1 << ": Index " << indices[i * k + j]
                      << " (Distance: " << distances[i * k + j] << ")\n";
        }
    }

    return 0;
}
