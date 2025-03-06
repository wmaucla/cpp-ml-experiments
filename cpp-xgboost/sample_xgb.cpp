#include <xgboost/c_api.h>
#include <iostream>
#include <vector>
#include <cstdlib> // for rand()

int main() {
    const int nrows = 10000;  // 10,000 training samples
    const int ncols = 10;     // 10 features per sample

    // Generate random training data
    std::vector<float> data(nrows * ncols);
    std::vector<float> labels(nrows);

    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            data[i * ncols + j] = static_cast<float>(rand()) / RAND_MAX;  // Random values between 0 and 1
        }
        labels[i] = (data[i * ncols] > 0.5) ? 1.0f : 0.0f;  // Simple rule-based labels
    }

    // Load training data into DMatrix
    DMatrixHandle dtrain;
    XGDMatrixCreateFromMat(data.data(), nrows, ncols, -1, &dtrain);
    XGDMatrixSetFloatInfo(dtrain, "label", labels.data(), nrows);

    // Create Booster
    BoosterHandle booster;
    XGBoosterCreate(&dtrain, 1, &booster);

    // Set model parameters
    XGBoosterSetParam(booster, "max_depth", "3");
    XGBoosterSetParam(booster, "eta", "0.1");
    XGBoosterSetParam(booster, "objective", "binary:logistic");

    // Train the model for 50 iterations
    for (int i = 0; i < 50; ++i) {
        XGBoosterUpdateOneIter(booster, i, dtrain);
    }

    // Generate random test data (1 sample)
    std::vector<float> test_data(ncols);
    for (int j = 0; j < ncols; ++j) {
        test_data[j] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Load test data into DMatrix
    DMatrixHandle dtest;
    XGDMatrixCreateFromMat(test_data.data(), 1, ncols, -1, &dtest);

    // Run prediction
    bst_ulong out_len = 0;  // Ensure this is initialized properly
    const float* out_result = nullptr;  // Ensure this is initialized properly
    int pred_success = XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);

    if (pred_success == 0 && out_len > 0 && out_result != nullptr) {
        std::cout << "Prediction for test sample: " << out_result[0] << std::endl;
    } else {
        std::cerr << "Error: Prediction failed." << std::endl;
    }

    // Cleanup
    XGBoosterFree(booster);
    XGDMatrixFree(dtrain);
    XGDMatrixFree(dtest);

    std::cout << "XGBoost training and prediction completed!" << std::endl;
    return 0;
}
