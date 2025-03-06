extern crate xgboost;
use xgboost::{parameters, DMatrix, Booster};
use std::time::Instant;

fn main() {
    let start_total = Instant::now(); // Start timing the whole process

    // Small training set
    let x_train = &[1.0, 1.0, 1.0,
                     1.0, 1.0, 0.0,
                     1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0,
                     1.0, 1.0, 1.0];
    let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];
    let num_train_rows = 5;

    let mut dtrain = DMatrix::from_dense(x_train, num_train_rows).unwrap();
    dtrain.set_labels(y_train).unwrap();

    let evaluation_sets = &[(&dtrain, "train")];

    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .unwrap();

    let bst = Booster::train(&training_params).unwrap();

    // Generate a large test dataset with 80M rows (same feature pattern)
    let num_test_rows = 80_000_000;
    let num_features = x_train.len() / num_train_rows;
    let x_test: Vec<f32> = x_train.iter()
        .copied()
        .cycle()
        .take(num_test_rows * num_features)
        .collect();

    let dtest = DMatrix::from_dense(&x_test, num_test_rows).unwrap();

    // Perform inference timing
    let start_inference = Instant::now();
    let preds = bst.predict(&dtest).unwrap();
    let inference_duration = start_inference.elapsed();

    println!("Inference over {} rows took: {:.2?}", num_test_rows, inference_duration);
    println!("First 10 predictions: {:?}", &preds[..10.min(preds.len())]);

    let total_duration = start_total.elapsed();
    println!("Total execution time: {:.2?}", total_duration);
}
