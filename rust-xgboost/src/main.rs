// use xgboost_rs::{DMatrix, Booster};
use xgboost_rs::{DMatrix};
use xgboost_rs::parameters::tree::TreeBoosterParametersBuilder;
// use xgboost_rs::parameters::BoosterParametersBuilder;
// use xgboost_rs::parameters::BoosterType;
// use xgboost_rs::learning::LearningTaskParametersBuilder;
// use xgboost_rs::learning::Objective;
// use xgboost_rs::learning::EvaluationMetric;
// use xgboost_rs::learning::Metrics;

fn main() {

    let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    let _dmat_train =
        DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
    let _dmat_test =
        DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

    let _tree_params = TreeBoosterParametersBuilder::default()
        .max_depth(2)
        .eta(1.0)
        .build()
        .unwrap();
    // let learning_params = learning::LearningTaskParametersBuilder::default()
    //     .objective(learning::Objective::BinaryLogistic)
    //     .eval_metrics(learning::Metrics::Custom(vec![
    //         learning::EvaluationMetric::MapCutNegative(4),
    //         learning::EvaluationMetric::LogLoss,
    //         learning::EvaluationMetric::BinaryErrorRate(0.5),
    //     ]))
    //     .build()
    //     .unwrap();
    // let params = parameters::BoosterParametersBuilder::default()
    //     .booster_type(parameters::BoosterType::Tree(tree_params))
    //     .learning_params(learning_params)
    //     .verbose(false)
    //     .build()
    //     .unwrap();
    // let mut booster =
    //     Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

    // for i in 0..10 {
    //     booster.update(&dmat_train, i).expect("update failed");
    // }

    // let eps = 1e-6;

    // let train_metrics = booster.evaluate(&dmat_train, "default").unwrap();
    // assert!(*train_metrics.get("logloss").unwrap() - 0.006_634 < eps);
    // assert!(*train_metrics.get("map@4-").unwrap() - 0.001_274 < eps);

    // let test_metrics = booster.evaluate(&dmat_test, "default").unwrap();
    // assert!(*test_metrics.get("logloss").unwrap() - 0.006_92 < eps);
    // assert!(*test_metrics.get("map@4-").unwrap() - 0.005_155 < eps);

    // let v = booster.predict(&dmat_test).unwrap();
    // assert_eq!(v.len(), dmat_test.num_rows());

    // // first 10 predictions
    // let expected_start = [
    //     0.005_015_169_3,
    //     0.988_446_7,
    //     0.005_015_169_3,
    //     0.005_015_169_3,
    //     0.026_636_455,
    //     0.117_893_63,
    //     0.988_446_7,
    //     0.012_314_71,
    //     0.988_446_7,
    //     0.000_136_560_63,
    // ];

    // // last 10 predictions
    // let expected_end = [
    //     0.002_520_344,
    //     0.000_609_179_26,
    //     0.998_810_05,
    //     0.000_609_179_26,
    //     0.000_609_179_26,
    //     0.000_609_179_26,
    //     0.000_609_179_26,
    //     0.998_110_2,
    //     0.002_855_195,
    //     0.998_110_2,
    // ];

    // for (pred, expected) in v.iter().zip(&expected_start) {
    //     assert!(pred - expected < eps);
    // }

    // for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
    //     assert!(pred - expected < eps);
    // }
}
