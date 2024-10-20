use egui::Slider;
use egui_plot::{Line, PlotResponse};
use nalgebra as na;

use crate::gp::RbfKernel;

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct App {
    x: Vec<f64>,
    y: Vec<f64>,
    kernel_length_scale: f64,
    kernel_sigma: f64,
    noise_sigma: f64,
    #[serde(skip)]
    gp: Option<crate::gp::GaussianProcess<RbfKernel>>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            x: vec![1.0, 2.0, 6.0],
            y: vec![1.0, 1.0, -1.0],
            kernel_sigma: 1.0,
            kernel_length_scale: 1.0,
            noise_sigma: 0.1,
            gp: None,
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for App {
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }

                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("Gaussian Processes");

            ui.label("Kernel parameters:");
            let mut changed = false;
            if ui
                .add(
                    Slider::new(&mut self.kernel_length_scale, 0.0..=10.0)
                        .text("Kernel length scale"),
                )
                .changed()
            {
                changed = true;
            }
            if ui
                .add(Slider::new(&mut self.kernel_sigma, 0.0..=10.0).text("Kernel sigma"))
                .changed()
            {
                changed = true;
            }
            if ui
                .add(Slider::new(&mut self.noise_sigma, 0.0..=10.0).text("Noise sigma"))
                .changed()
            {
                changed = true;
            }

            ui.label("Click anywhere to add points, click on points to remove them.");
            if ui.button("Clear all Points").clicked() {
                self.x.clear();
                self.y.clear();
                changed = true;
            }
            ui.label("Ctrl-Scroll to zoom, Scroll and Shift-scroll to pan.");

            if let Some(gp) = &self.gp {
                // linearly spaced points from 0 to 10
                let prediction_x = (0..=100)
                    .map(|i| i as f64 / 100.0 * 10.0)
                    .collect::<Vec<f64>>();

                let (means, variances) = gp.predict(&na::DVector::from_vec(prediction_x.clone()));

                let mean_points: egui_plot::PlotPoints = means
                    .iter()
                    .zip(prediction_x.iter())
                    .map(|(y, x)| [*x, *y])
                    .collect();
                let mean_line = egui_plot::Line::new(mean_points).color(egui::Color32::RED);

                // egui_plot does not support filling non-convex polygons, so we fallback to
                // drawing some lines to represent the variance instead.

                // lower variance points
                let variance_points = variances
                    .iter()
                    .zip(means.iter())
                    .zip(prediction_x.iter())
                    .map(|((sigma, mean), x)| [*x, (*mean - *sigma)])
                    .collect::<Vec<[f64; 2]>>();
                let lower_variance_line =
                    Line::new(variance_points).color(egui::Color32::LIGHT_BLUE);

                // upper variance points
                let variance_points = variances
                    .iter()
                    .zip(means.iter())
                    .zip(prediction_x.iter())
                    .map(|((sigma, mean), x)| [*x, (*mean + *sigma)])
                    .collect::<Vec<[f64; 2]>>();
                let upper_variance_line =
                    Line::new(variance_points).color(egui::Color32::LIGHT_BLUE);

                // the points the GP was trained on
                let points: egui_plot::PlotPoints = self
                    .x
                    .iter()
                    .zip(self.y.iter())
                    .map(|(x, y)| [*x, *y])
                    .collect();
                let points = egui_plot::Points::new(points)
                    .color(egui::Color32::LIGHT_GREEN)
                    .radius(5.0)
                    .shape(egui_plot::MarkerShape::Circle)
                    .id(egui::Id::new("training_points"));

                let PlotResponse {
                    response: _,
                    inner: (pointer_coordinate, clicked),
                    hovered_plot_item,
                    ..
                } = egui_plot::Plot::new("plot").show(ui, |pui| {
                    pui.line(lower_variance_line.name("Mean - Variance"));
                    pui.line(upper_variance_line.name("Mean + Variance"));
                    pui.line(mean_line.name("Mean"));
                    pui.points(points.name("Training points"));
                    (pui.pointer_coordinate(), pui.response().clicked())
                });

                if clicked {
                    if let (Some(hovered_plot_item), Some(pos)) =
                        (hovered_plot_item, pointer_coordinate)
                    {
                        if hovered_plot_item == egui::Id::new("training_points") {
                            // find the index of the point that was clicked

                            if let Some((index, _)) = self
                                .x
                                .iter()
                                .zip(self.y.iter())
                                .map(|(x, y)| (*x - pos.x).powf(2.0) + (*y - pos.y).powf(2.0))
                                .enumerate()
                                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            {
                                self.x.remove(index);
                                self.y.remove(index);
                                changed = true;
                            }
                        }
                    } else if let Some(pointer_coordinate) = pointer_coordinate {
                        self.x.push(pointer_coordinate.x);
                        self.y.push(pointer_coordinate.y);
                        changed = true;
                    }
                }
            }

            if changed || self.gp.is_none() {
                self.gp = Some(crate::gp::GaussianProcess::new(
                    &na::DVector::from_vec(self.x.clone()),
                    &na::DVector::from_vec(self.y.clone()),
                    RbfKernel {
                        sigma: self.kernel_sigma,
                        length_scale: self.kernel_length_scale,
                    },
                    self.noise_sigma,
                ));
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                powered_by_egui_and_eframe(ui);
                egui::warn_if_debug_build(ui);
            });
        });
    }
}

fn powered_by_egui_and_eframe(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.label("Powered by ");
        ui.hyperlink_to("egui", "https://github.com/emilk/egui");
        ui.label(" and ");
        ui.hyperlink_to(
            "eframe",
            "https://github.com/emilk/egui/tree/master/crates/eframe",
        );
        ui.label(".");
    });
}
