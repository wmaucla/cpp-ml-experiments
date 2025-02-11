use catboost;

fn main() {
    let model_path = "model.cbm";
    let model = catboost::Model::load(model_path).unwrap();

    println!("Adult dataset model metainformation\n");

    println!("tree count: {}", model.get_tree_count());

    println!("prediction dimension: {}", model.get_dimensions_count());

    println!("numeric feature count: {}", model.get_float_features_count());

}