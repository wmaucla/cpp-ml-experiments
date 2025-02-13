extern crate xgboost;
use xgboost::{parameters, DMatrix, Booster};

fn main() {
    // training matrix with 5 training examples and 3 features
    let x_train = &[1.0, 1.0, 1.0,
                     1.0, 1.0, 0.0,
                     1.0, 1.0, 1.0,
                     0.0, 0.0, 0.0,
                     1.0, 1.0, 1.0];
    let num_rows = 5;
    let y_train = &[1.0, 1.0, 1.0, 0.0, 1.0];

    let mut dtrain = DMatrix::from_dense(x_train, num_rows).unwrap();
    dtrain.set_labels(y_train).unwrap();

    let x_test = &[0.7, 0.9, 0.6];
    let num_rows = 1;
    let y_test = &[1.0];
    let mut dtest = DMatrix::from_dense(x_test, num_rows).unwrap();
    dtest.set_labels(y_test).unwrap();
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    let training_params = parameters::TrainingParametersBuilder::default()
    .dtrain(&dtrain)
    .evaluation_sets(Some(evaluation_sets))
    .build()
    .unwrap();

    let bst = Booster::train(&training_params).unwrap();

    // Get predictions probabilities for given matrix (as ndarray::Array1)
    let preds = bst.predict(&dtest).unwrap();
    
    println!("{:?}", bst.predict(&dtest).unwrap());

    // Get predicted labels for each test example (0.0 or 1.0 in this case)
    let labels = dtest.get_labels().unwrap();

    // Print error rate
    let mut num_errors = 0;
    for (pred, label) in preds.iter().zip(labels) {
        let pred = if *pred > 0.5 { 1.0 } else { 0.0 };
        if pred != *label {
            num_errors += 1;
        }
    }
    println!("error={} ({}/{} correct)",
             num_errors as f32 / preds.len() as f32, preds.len() - num_errors, preds.len());
}