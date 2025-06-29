use plotters::prelude::*;
use std::io::Write;
use image::{ImageBuffer, Rgb};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 640;
    let height = 480;
    let mut buffer = vec![0u8; width * height * 3]; // 3 bytes per pixel (RGB)

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

    // Create ImageBuffer with RGB format
    let img_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_vec(width as u32, height as u32, buffer)
        .expect("Failed to create image buffer from vector");

    // // Debug save
    // img_buffer.save("debug_buffer.png").expect("Failed to save debug image");

    // Encode to PNG
    let mut png_buffer = Vec::new();
    img_buffer.write_to(&mut std::io::Cursor::new(&mut png_buffer), image::ImageFormat::Png)?;

    // Write to stdout
    let mut stdout = std::io::stdout();
    stdout.write_all(&png_buffer)?;
    stdout.flush()?;

    Ok(())
}
