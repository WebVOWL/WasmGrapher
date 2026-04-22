//! Event channels for communicating with the outside GUI world.

use glam::Vec2;

/// Describes an event received by a GUI.
#[derive(Clone, Copy)]
pub enum GUIEvent {
    /// Show metadata for element with this index.
    ShowMetadata(usize),

    /// Metadata of an element is no longer shown.
    HideMetadata(),
}
