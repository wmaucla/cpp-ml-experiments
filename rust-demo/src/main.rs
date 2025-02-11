use linfa_trees::DecisionTree;
use linfa::prelude::*;
use linfa_datasets;

fn main() {
    let dataset = linfa_datasets::iris();
    let tree = DecisionTree::params().fit(&dataset).unwrap();
    let accuracy = tree.predict(&dataset).confusion_matrix(&dataset).unwrap().accuracy();
    println!("Model Accuracy: {:.2}%", accuracy * 100.0);
    assert!(accuracy > 0.9);
}
