use plotters::prelude::*;
use std::io::Write;
use image::{ImageBuffer, Rgba};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 640;
    let height = 480;
    let mut buffer = vec![0u8; width * height * 4];

    {
        let root = BitMapBackend::with_buffer(&mut buffer, (width as u32, height as u32)).into_drawing_area();
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

    // Transpose the buffer (treat it as height x width instead of width x height)
    let mut corrected_buffer = Vec::with_capacity(buffer.len());
    for y in 0..height {
        for x in 0..width {
            let orig_idx = (x * height + y) * 4; // Treat as column-major
            if orig_idx + 3 < buffer.len() {
                corrected_buffer.extend_from_slice(&buffer[orig_idx..orig_idx + 4]);
            }
        }
    }

    // Create ImageBuffer with original dimensions
    let img_buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_vec(width as u32, height as u32, corrected_buffer)
        .expect("Failed to create image buffer from vector");

    // Debug save
    img_buffer.save("debug_buffer.png").expect("Failed to save debug image");

    // Encode to PNG
    let mut png_buffer = Vec::new();
    img_buffer.write_to(&mut std::io::Cursor::new(&mut png_buffer), image::ImageFormat::Png)?;

    // Write to stdout
    let mut stdout = std::io::stdout();
    stdout.write_all(&png_buffer)?;
    stdout.flush()?;

    Ok(())
}
