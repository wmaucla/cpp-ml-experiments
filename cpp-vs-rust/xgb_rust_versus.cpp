#include <xgboost/c_api.h>
#include <vector>
#include <iostream>
#include <chrono>

int main() {
    auto start_total = std::chrono::high_resolution_clock::now(); // Start total timing

    // Small training set (5 rows, 3 features)
    std::vector<float> x_train = {1.0, 1.0, 1.0,
                                  1.0, 1.0, 0.0,
                                  1.0, 1.0, 1.0,
                                  0.0, 0.0, 0.0,
                                  1.0, 1.0, 1.0};
    std::vector<float> y_train = {1.0, 1.0, 1.0, 0.0, 1.0};
    int num_train_rows = 5;
    int num_features = 3;

    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(x_train.data(), num_train_rows, num_features, -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", y_train.data(), num_train_rows);

    // Training parameters
    BoosterHandle booster;
    DMatrixHandle eval_dmats[1] = {dtrain};
    const char *eval_names[1] = {"train"};
    XGBoosterCreate(eval_dmats, 1, &booster);
    XGBoosterSetParam(booster, "objective", "binary:logistic");
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "eval_metric", "logloss");

    // Train for 10 rounds
    for (int iter = 0; iter < 10; ++iter) {
        XGBoosterUpdateOneIter(booster, iter, dtrain);
    }

    // Generate a large test dataset (80M rows, same pattern)
    int num_test_rows = 80'000'000;
    std::vector<float> x_test(num_test_rows * num_features);
    for (size_t i = 0; i < x_test.size(); i++) {
        x_test[i] = x_train[i % x_train.size()]; // Repeat pattern
    }

    DMatrixHandle dtest;
    XGDMatrixCreateFromMat(x_test.data(), num_test_rows, num_features, -1, &dtest);

    // Perform inference and measure time
    auto start_inference = std::chrono::high_resolution_clock::now();
    bst_ulong out_len;
    const float *out_result;  // Fix type to `const float *`
    XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);
    auto inference_duration = std::chrono::high_resolution_clock::now() - start_inference;

    std::cout << "Inference over " << num_test_rows << " rows took: "
              << std::chrono::duration<double>(inference_duration).count() << " seconds\n";

    // Print first 10 predictions
    std::cout << "First 10 predictions: ";
    for (size_t i = 0; i < std::min(static_cast<bst_ulong>(out_len), static_cast<bst_ulong>(10)); i++) {
        std::cout << out_result[i] << " ";
    }
    std::cout << "\n";

    // Measure total execution time
    auto total_duration = std::chrono::high_resolution_clock::now() - start_total;
    std::cout << "Total execution time: "
              << std::chrono::duration<double>(total_duration).count() << " seconds\n";

    // Cleanup
    XGDMatrixFree(dtrain);
    XGDMatrixFree(dtest);
    XGBoosterFree(booster);

    return 0;
}
