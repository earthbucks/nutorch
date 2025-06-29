use image::{DynamicImage, ImageBuffer, Rgb, Rgba};
use plotters::prelude::*;
use viuer::{print, Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 800;
    let height = 800;
    let mut buffer = vec![0u8; width * height * 4]; // 4 bytes per pixel (RGBA)

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width as u32, height as u32))
            .into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("y=x^2", ("sans-serif", 50).into_font())
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(-1f32..1f32, -0.1f32..1f32)?;

        chart.configure_mesh().draw()?;

        chart
            .draw_series(LineSeries::new(
                (-50..=50).map(|x| x as f32 / 50.0).map(|x| (x, x * x)),
                &RED,
            ))?
            .label("y = x^2")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        root.present()?;
    }

    // Create ImageBuffer with RGB format
    let rgb_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, buffer)
        .expect("Failed to create RGB image buffer from vector");

    // Convert RGB to RGBA by adding alpha channel (set to 255 for opacity)
    let mut rgba_data = Vec::with_capacity(width * height * 4);
    for pixel in rgb_buffer.pixels() {
        rgba_data.extend_from_slice(&[pixel.0[0], pixel.0[1], pixel.0[2], 255]);
        // R, G, B, A
    }
    let rgba_buffer =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width as u32, height as u32, rgba_data)
            .expect("Failed to create RGBA image buffer");

    // Configure viuer for terminal display
    let conf = Config {
        width: Some(160), // Adjust based on terminal width; optional
        height: Some(80), // Adjust based on terminal height; optional
        use_kitty: true,
        use_iterm: true,
        use_sixel: true,
        truecolor: true,
        x: 20,
        y: 4,
        restore_cursor: false,
        ..Default::default()
    };

    // Convert ImageBuffer to DynamicImage
    let dynamic_img: DynamicImage = DynamicImage::ImageRgba8(rgba_buffer);

    // Display the image in the terminal using viuer (implicit conversion to DynamicImage)
    print(&dynamic_img, &conf).expect("Failed to display image in terminal");

    // Alternative: Explicit conversion to DynamicImage if needed (uncomment if implicit fails)
    // let dynamic_img = image::DynamicImage::ImageRgba8(img_buffer);
    // print(&dynamic_img, &conf).expect("Failed to display image in terminal");

    // Optional: Still save for debugging
    rgb_buffer
        .save("debug_buffer.png")
        .expect("Failed to save debug image");

    Ok(())
}
