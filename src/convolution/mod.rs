
use std::ops::Index;
use std::f64;

#[derive(Debug)]
pub enum ConvolutionError {
    ConvolutionIsNotWellShaped,
    ConvolutionDoesNotContainNumbers,
}

fn is_correctly_dimensionsed(convolution: &[f64]) -> bool {
    let sqrt = (convolution.len() as f64).sqrt();
    let is_perfect_square = (sqrt.floor() * sqrt.floor()) as usize == convolution.len();
    let is_odd_dimensionsed = sqrt as u64 % 2 == 1;

    is_perfect_square && is_odd_dimensionsed
}

pub struct Convolution<'a> {
    convolution: &'a [f64],
    size: usize,
    min_value: f64,
    max_value: f64,
}

impl<'a> Convolution<'a> {
    pub fn new(values: &'a [f64]) -> Result<Convolution<'a>, ConvolutionError> {
        if values.iter().cloned().any(
            |x| x.is_infinite() || x.is_nan(),
        )
        {
            return Err(ConvolutionError::ConvolutionDoesNotContainNumbers);
        }
        if !is_correctly_dimensionsed(values) {
            return Err(ConvolutionError::ConvolutionIsNotWellShaped);
        }

        let convolution_size = (values.len() as f64).sqrt().floor() as usize;

        let min_value = if any_negative(values) {
            sum_of_negatives(values)
        } else {
            minimal_positive(values)
        };

        let max_value = if any_positive(values) {
            sum_of_positives(values)
        } else {
            maximal_negative(values)
        };

        Ok(Convolution {
            convolution: values,
            size: convolution_size,
            min_value: min_value,
            max_value: max_value,
        })
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    #[allow(float_cmp)]
    pub fn compute_adjusted_pixel_value(&self, value: f64) -> u8 {
        if self.max_value == self.min_value {
            value as u8
        } else {
            ((value - self.min_value) / (self.max_value - self.min_value)) as u8
        }
    }
}

impl<'a> Index<(usize, usize)> for Convolution<'a> {
    type Output = f64;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.convolution[(index.1 * self.size) + index.0]
    }
}

fn any_negative(values: &[f64]) -> bool {
    values.iter().any(|&x| x < 0.0)
}

fn any_positive(values: &[f64]) -> bool {
    values.iter().any(|&x| x >= 0.0)
}

fn sum_of_positives(values: &[f64]) -> f64 {
    values.iter().cloned().filter(|&x| x >= 0.0).sum()
}

fn sum_of_negatives(values: &[f64]) -> f64 {
    values.iter().cloned().filter(|&x| x < 0.0).sum()
}

fn minimal_positive(values: &[f64]) -> f64 {
    values
        .iter()
        .cloned()
        .filter(|&x| x >= 0.0)
        .min_by(|&x, &y| x.partial_cmp(&y).unwrap())
        .unwrap()
}

fn maximal_negative(values: &[f64]) -> f64 {
    values
        .iter()
        .cloned()
        .filter(|&x| x < 0.0)
        .max_by(|&x, &y| x.partial_cmp(&y).unwrap())
        .unwrap()
}
