use std::panic::Location;

#[derive(Debug)]
pub enum WasmGrapherErrorKind {
    /// Errors related to the renderer.
    RenderError(String),
    /// Errors related to the graph simulator.
    SimulatorError(String),
    /// Errors related to event handling.
    EventHandlerError(String),
}

#[derive(Debug)]
pub struct WasmGrapherError {
    /// The contained error type.
    inner: WasmGrapherErrorKind,
    /// The error's location in the source code.
    location: &'static Location<'static>,
}
impl std::fmt::Display for WasmGrapherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.inner)
    }
}

impl From<WasmGrapherErrorKind> for WasmGrapherError {
    #[track_caller]
    fn from(error: WasmGrapherErrorKind) -> Self {
        WasmGrapherError {
            inner: error,
            location: Location::caller(),
        }
    }
}
