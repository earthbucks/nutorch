use plotters::prelude::*;
use std::io::Write;
use image::{ImageBuffer, Rgba};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a buffer to store the raw image data in memory
    let mut buffer = vec![0u8; 640 * 480 * 4]; // RGBA format (4 bytes per pixel)

    // Scope to limit the lifetime of root and associated references
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (640, 480)).into_drawing_area();
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
    } // root is dropped here, releasing the mutable borrow on buffer

    // Now buffer can be used since root is dropped
    let img_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(640, 480, buffer)
        .expect("Failed to create image buffer from vector");

    // Encode the image to PNG in a temporary in-memory buffer
    let mut png_buffer = Vec::new();
    img_buffer.write_to(&mut std::io::Cursor::new(&mut png_buffer), image::ImageFormat::Png)?;

    // Write the PNG data to stdout
    let mut stdout = std::io::stdout();
    stdout.write_all(&png_buffer)?;
    stdout.flush()?;

    Ok(())
}
