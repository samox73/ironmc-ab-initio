//! Model-agnostic imaginary-time diagrammatic Monte Carlo.

pub mod diagram;
pub mod measure;
pub mod model;
pub mod run;
pub mod updates;
pub mod weight;

pub use diagram::{Diagram, Vertex};
pub use measure::{summarize, DiagMeasurement, Estimate, GreenFunction, MeasurementData, Report};
pub use model::{FrohlichModel, Model};
pub use run::{run, run_leg, DiagConfig, SlotmapLeg};
pub use updates::{update_set, update_set_fixed_tau, DiagUpdate};
