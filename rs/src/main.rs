use tch::{Device, Kind, Tensor};

fn linspace(start: f64, end: f64, steps: i64) -> Tensor {
    if steps < 2 {
        panic!("Number of steps must be at least 2");
    }
    Tensor::linspace(start, end, steps, (Kind::Float, Device::Cpu))
}

fn sin(tensor: &Tensor) -> Tensor {
    tensor.sin()
}

fn main() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
    let t = linspace(0.0, 10.0, 10);
    t.print();
    let t = sin(&t);
    t.print();
}
