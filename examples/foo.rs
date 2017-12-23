extern crate convolve;
extern crate image;

use convolve::{Convolution, convolve, EdgeMode};
use std::time::Instant;


fn main() {
    let base_image = image::open("img/sam.jpg").expect("sam.jpg failed to open");

    let gaussian_convoloution = Convolution::new(
        &[
            0.003765,
            0.015019,
            0.023792,
            0.015019,
            0.003765,
            0.015019,
            0.059912,
            0.094907,
            0.059912,
            0.015019,
            0.023792,
            0.094907,
            0.150342,
            0.094907,
            0.023792,
            0.015019,
            0.059912,
            0.094907,
            0.059912,
            0.015019,
            0.003765,
            0.015019,
            0.023792,
            0.015019,
            0.003765,
        ],
    ).expect("making a convolution");

    let start = Instant::now();
    let mut build = Vec::with_capacity(1000);
    for i in 0..1000 {
        let new_image = base_image.clone();
        let img = convolve(new_image, &gaussian_convoloution, EdgeMode::Extend)
            .expect("convolving");
        let k = i % 20;
        build.push(img.get_pixel(3, k).data[0]);
    }
    let end = Instant::now();
    println!("{:?}", build);

    println!(
        "{}",
        1000.0 / ((end - start).as_secs() as f64 + ((end - start).subsec_nanos() as f64 / 1e9))
    );
}
