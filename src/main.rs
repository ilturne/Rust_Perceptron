use rand::Rng;
use plotters::prelude::*;

pub struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}

fn main() {
    let mut rng = rand::thread_rng();
    let num_features = 2;
    let learning_rate = 0.1;
    let epochs = 1000000;

    // Generate clustered data for training
    let cluster_size = 20;
    let mut training_data = vec![];

    // Define cluster centers
    let center1 = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));
    let center2 = (rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0));

    // Generate data around cluster centers
    for _ in 0..cluster_size {
        let x1: f64 = rng.gen_range(center1.0 - 0.5..center1.0 + 0.5);
        let x2: f64 = rng.gen_range(center1.1 - 0.5..center1.1 + 0.5);
        training_data.push((vec![x1, x2], false));
    }
    for _ in 0..cluster_size {
        let x1: f64 = rng.gen_range(center2.0 - 0.5..center2.0 + 0.5);
        let x2: f64 = rng.gen_range(center2.1 - 0.5..center2.1 + 0.5);
        training_data.push((vec![x1, x2], true));
    }

    let mut perceptron = Perceptron::new(num_features, learning_rate);
    perceptron.train(&training_data, epochs);

    if perceptron.can_separate(&training_data) {
        plot_data(&training_data, &perceptron);
    } else {
        println!("A line cannot be made.");
    }
}

// This will be our activation function which is a step function
fn step_function(input: f64) -> bool {
    input > 0.0
}

impl Perceptron {
    // This initializes random weights and bias for the Perceptron. The num_features is just the range of random numbers used
    pub fn new(num_features: usize, learning_rate: f64) -> Perceptron {
        let mut rng = rand::thread_rng();
        let weights = (0..num_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias = rng.gen_range(-1.0..1.0);
        Perceptron {
            weights,
            bias,
            learning_rate,
        }
    }

    fn predict(&self, inputs: &Vec<f64>) -> bool {
        let sum: f64 = self.weights.iter().zip(inputs).map(|(w, x)| w * x).sum::<f64>() + self.bias;
        step_function(sum)
    }

    fn train(&mut self, training_data: &Vec<(Vec<f64>, bool)>, epochs: usize) {
        for _ in 0..epochs {
            for (inputs, target) in training_data {
                let prediction = self.predict(inputs);
                let error = (*target as i32 - prediction as i32) as f64;

                for i in 0..self.weights.len() {
                    self.weights[i] += self.learning_rate * error * inputs[i];
                }
                self.bias += self.learning_rate * error;
            }
        }
    }

    fn can_separate(&self, training_data: &Vec<(Vec<f64>, bool)>) -> bool {
        training_data.iter().all(|(inputs, target)| self.predict(inputs) == *target)
    }
}

fn plot_data(training_data: &Vec<(Vec<f64>, bool)>, perceptron: &Perceptron) {
    let root = BitMapBackend::new("plot.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Perceptron Decision Boundary", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Plot the data points
    for (inputs, target) in training_data {
        let color = if *target { &RED } else { &BLUE };
        chart
            .draw_series(PointSeries::of_element(
                vec![(inputs[0], inputs[1])],
                5,
                color,
                &|c, s, st| {
                    return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
                        + Circle::new((0, 0), s, st.filled());
                },
            ))
            .unwrap();
    }

    // Plot the decision boundary
    let slope = -perceptron.weights[0] / perceptron.weights[1];
    let intercept = -perceptron.bias / perceptron.weights[1];
    chart
        .draw_series(LineSeries::new(
            vec![(-1.0, slope * -1.0 + intercept), (1.0, slope * 1.0 + intercept)],
            &BLACK,
        ))
        .unwrap();
}
