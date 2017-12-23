#![deny(warnings)]
extern crate image;

mod convolution;

pub use convolution::{Convolution, ConvolutionError};

use image::{DynamicImage, GrayImage, ImageBuffer};

pub enum EdgeMode {
    Extend,
    Wrap,
    Mirror,
}

pub fn convolve(image_in: DynamicImage, convolution: Convolution, edge_mode: EdgeMode) -> Result<GrayImage, ConvolutionError> {
    let gray_image = image_in.to_luma();
    let mut build = Vec::with_capacity((gray_image.width() * gray_image.height()) as usize);

    for (x, y, _) in gray_image.enumerate_pixels() {
        let mut this_pixel = 0.0;
        let offset = (convolution.get_size() as i64) /2;
        for y_offset in -offset..offset+1 {
            for x_offset in -offset..offset+1 {
                let p = get_pixel(&gray_image, x as usize + x_offset as usize, y as usize + y_offset as usize, &edge_mode);
                this_pixel += p as f64 * convolution[((x_offset + offset) as usize, (y_offset + offset) as usize)];
            }

        }

        build.push(convolution.compute_adjusted_pixel_value(this_pixel));
    }

    Ok(ImageBuffer::from_raw(gray_image.width(), gray_image.height(), build).unwrap())
}

fn get_pixel(image: &image::GrayImage, x: usize, y: usize, edge_mode: &EdgeMode) -> u8 {
    let (proj_x, proj_y) = edge_project(x, y, image.width(), image.height(), edge_mode);

    return image.get_pixel(proj_x, proj_y).data[0]
}

fn edge_project(x: usize, y: usize, width: u32, height: u32, edge_mode: &EdgeMode) -> (u32, u32) {
    if x as u32 <= width-1 && y as u32 <= height-1 {
        return (x as u32, y as u32)
    }

    let x = x as i64;
    let y = y as i64;
    let width = width as i64;
    let height = height as i64;

    match edge_mode {
        &EdgeMode::Extend => {
            let ret_x = if x < 0 {
                0
            } else if x > width-1 {
                width-1
            } else {
                unreachable!()
            };


            let ret_y = if y < 0 {
                0
            } else if y > height-1 {
                height -1
            } else {
                unreachable!()
            };

            (ret_x as u32, ret_y as u32)
        },
        &EdgeMode::Wrap => {
            let ret_x = if x < 0 {
                x + width
            } else if x > width-1 {
                x - width
            } else {
                unreachable!()
            };


            let ret_y = if y < 0 {
                y + height
            } else if y > height-1 {
                y - height
            } else {
                unreachable!()
            };

            (ret_x as u32, ret_y as u32)
        },
        &EdgeMode:: Mirror => {
            let ret_x = if x < 0 {
                -x - 1
            } else if x > width-1 {
                -x + (2*width) -1
            } else {
                unreachable!()
            };


            let ret_y = if y < 0 {
                -y - 1
            } else if y > height-1 {
                -y + (2*height) -1
            } else {
                unreachable!()
            };

            (ret_x as u32, ret_y as u32)
        }
    }
}

#[cfg(test)]
mod tests {

    #[derive(PartialEq, Eq, Debug)]
    struct TestableGrayImg {
        pixels: Vec<u8>
    }

    use super::{convolve, EdgeMode, Convolution};
    use image::{self, GrayImage};
    #[test]
    fn identity_convolution_returns_same_image() {
        let base_image = image::open("img/sam.jpg").expect("sam.jpg failed to open");
        let identity_convoloution = Convolution::new(&[ 1.0 ]).expect("making a convolution");

        let gray_base = base_image.clone().to_luma();

        assert_eq!(
            testable_repr(convolve(base_image, identity_convoloution, EdgeMode::Extend).expect("unwrapping image")),
            testable_repr(gray_base)
        );
    }

    fn testable_repr(img: GrayImage) -> TestableGrayImg {
        let mut build = Vec::with_capacity((img.width() * img.height()) as usize);
        for pixel in img.pixels() {
            build.push(pixel.data[0]);
        }

        TestableGrayImg {
            pixels: build,
        }
    }
}
