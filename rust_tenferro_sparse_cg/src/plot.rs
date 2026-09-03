use std::error::Error;

use plotters::prelude::*;

fn idx(i: usize, j: usize, n: usize) -> usize {
    i + n * j
}

fn grid_lookup(data: &[f64], n: usize, h: f64, x: f64, y: f64) -> f64 {
    let i = ((x / h).round() as usize).min(n - 1);
    let j = ((y / h).round() as usize).min(n - 1);
    data[idx(i, j, n)]
}

fn surface_style(z: &f64) -> ShapeStyle {
    HSLColor(0.65 * (1.0 - *z), 0.7, 0.45).filled()
}

fn viridis(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let stops = [
        (0.0, [68.0, 1.0, 84.0]),
        (0.25, [59.0, 82.0, 139.0]),
        (0.5, [33.0, 145.0, 140.0]),
        (0.75, [94.0, 201.0, 98.0]),
        (1.0, [253.0, 231.0, 37.0]),
    ];
    let mut index = 0;
    while index + 1 < stops.len() && t > stops[index + 1].0 {
        index += 1;
    }
    let (t0, c0) = stops[index];
    let (t1, c1) = stops[(index + 1).min(stops.len() - 1)];
    let weight = if (t1 - t0).abs() < 1.0e-12 {
        0.0
    } else {
        (t - t0) / (t1 - t0)
    };
    RGBColor(
        (c0[0] + weight * (c1[0] - c0[0])).round() as u8,
        (c0[1] + weight * (c1[1] - c0[1])).round() as u8,
        (c0[2] + weight * (c1[2] - c0[2])).round() as u8,
    )
}

pub fn save_plot(
    path: &str,
    coordinates: &[f64],
    solution: &[f64],
    exact: &[f64],
    error: &[f64],
    grid_size: usize,
    h: f64,
    max_error: f64,
) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new(path, (2400, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(4, 12, 4, 28);
    let (left, rest) = root.split_horizontally(740);
    let (middle, right) = rest.split_horizontally(740);

    let draw_surface = |area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
                        title: &str,
                        values: &[f64]|
     -> Result<(), Box<dyn Error>> {
        let mut chart = ChartBuilder::on(area)
            .caption(title, ("sans-serif", 18))
            .margin(8)
            .build_cartesian_3d(0.0..1.0, 0.0..1.05, 0.0..1.0)?;
        chart.with_projection(|mut projection| {
            projection.pitch = 30.0_f64.to_radians();
            projection.yaw = 45.0_f64.to_radians();
            projection.scale = 0.85;
            projection.into_matrix()
        });
        chart
            .configure_axes()
            .light_grid_style(BLACK.mix(0.15))
            .draw()?;
        chart.draw_series(
            SurfaceSeries::xoz(
                coordinates.iter().copied(),
                coordinates.iter().copied(),
                |x, y| grid_lookup(values, grid_size, h, x, y),
            )
            .style_func(&surface_style),
        )?;
        Ok(())
    };

    draw_surface(&left, "Exact solution", exact)?;
    draw_surface(&middle, "Numerical solution", solution)?;

    let (heatmap_area, colorbar_area) = right.split_horizontally(700);
    let mut chart = ChartBuilder::on(&heatmap_area)
        .caption("Absolute error", ("sans-serif", 18))
        .margin(8)
        .x_label_area_size(28)
        .y_label_area_size(36)
        .build_cartesian_2d(0.0..1.0, 0.0..1.0)?;
    chart.configure_mesh().x_desc("x").y_desc("y").draw()?;

    let scale = if max_error > 0.0 { max_error } else { 1.0 };
    chart.draw_series((0..grid_size).flat_map(|i| {
        (0..grid_size).map(move |j| {
            let color = viridis((error[idx(i, j, grid_size)] / scale).clamp(0.0, 1.0));
            Rectangle::new(
                [
                    (coordinates[i], coordinates[j]),
                    (coordinates[i] + h, coordinates[j] + h),
                ],
                color.filled(),
            )
        })
    }))?;

    let mut colorbar = ChartBuilder::on(&colorbar_area)
        .caption("|u - u_exact|", ("sans-serif", 14))
        .margin_left(8)
        .margin_right(28)
        .margin_top(28)
        .margin_bottom(36)
        .y_label_area_size(72)
        .build_cartesian_2d(0.0..1.0, 0.0..scale)?;
    colorbar
        .configure_mesh()
        .disable_x_mesh()
        .disable_x_axis()
        .y_label_formatter(&|value| format!("{value:.1e}"))
        .draw()?;

    const COLOR_BARS: usize = 64;
    colorbar.draw_series((0..COLOR_BARS).map(|index| {
        let y0 = scale * index as f64 / COLOR_BARS as f64;
        let y1 = scale * (index + 1) as f64 / COLOR_BARS as f64;
        let color = viridis((index as f64 + 0.5) / COLOR_BARS as f64);
        Rectangle::new([(0.0, y0), (1.0, y1)], color.filled())
    }))?;

    root.present()?;
    Ok(())
}
