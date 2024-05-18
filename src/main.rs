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
    let epochs = 1000;

    //Generate some random data for the traning
    let training_data: Vec<(Vec<f64>, bool)> = (0..100).map(|_| {
        let x1: f64 = rng.gen_range(-1.0..1.0);
        let x2: f64 = rng.gen_range(-1.0..1.0);
        let y = if x1 + x2 > 0.0 { true } else { false };
        (vec![x1, x2], y)
    }).collect();

    let mut perceptron = Perceptron::new(num_features, learning_rate);
    perceptron.train(&training_data, epochs);

    plot_data(&training_data, &perceptron);
}

//This will be our activation function which is a step function
fn step_function(input: f64) -> bool {
    return if input > 0.0 {
        true
    } else {
        false
    }
}

impl Perceptron {
    //This initializes random weights and bias for the Perceptron. The num_features is just the range of random numbers used
    pub fn new(num_features: usize, learning_rate: f64) -> Perceptron {
        let mut rng = rand::thread_rng();
        let weights = (0..num_features).map(|_| rng.gen_range(-1.0..1.0)).collect(); //The map function is applied to the vector and iterates through each item in the vector. The |_| is used  to indicate that a variable is intentionally ignored (it's random, so we don't care what the value is).
        let bias = rng.gen_range(-1.0..1.0);
        Perceptron {
            weights,
            bias,
            learning_rate,
        }
    }

    fn predict(&self, inputs: &Vec<f64>) -> bool {
        let sum: f64 = self.weights.iter().zip(inputs).map(|(w,x)| w * x).sum::<f64>() + self.bias;
        return step_function(sum);
    }


    fn train(&mut self, training_data: &Vec<(Vec<f64>, bool)>, epochs: usize) {
        /* An epoch represents one complete pass through the entire training dataset. */
        for _ in 0..epochs {
            for (inputs, target) in training_data {
                let prediction = self.predict(inputs);
                let error = (*target as i32 - prediction as i32) as f64;

                /*This inner loop iterates over each data point in the training dataset.*/
                for i in 0..self.weights.len() {
                    self.weights[i] += self.learning_rate * error * inputs[i];

                }
                self.bias += self.learning_rate * error;
            }
        }
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
                        + Circle::new((0,0),s,st.filled());
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