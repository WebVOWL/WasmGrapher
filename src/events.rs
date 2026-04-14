pub mod gui_events;
pub mod render_event;
pub mod simulator_event;

use flume::{Receiver, Sender};

use crate::events::{
    gui_events::GUIEvent, render_event::RenderEvent, simulator_event::SimulatorEvent,
};

#[expect(clippy::struct_field_names)]
pub struct EventDispatcher {
    /// Receiver must only be consumed by the simulator
    pub sim_read_chan: Receiver<SimulatorEvent>,
    pub sim_write_chan: Sender<SimulatorEvent>,
    /// Receiver must only be consumed by the renderer
    pub rend_read_chan: Receiver<RenderEvent>,
    pub rend_write_chan: Sender<RenderEvent>,
    /// Receiver must only be consumed by the GUI
    pub gui_read_chan: Receiver<GUIEvent>,
    pub gui_write_chan: Sender<GUIEvent>,
}

impl Default for EventDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl EventDispatcher {
    pub fn new() -> Self {
        let (ssc, src) = flume::unbounded();
        let (rsc, rrc) = flume::unbounded();
        let (gsc, grc) = flume::unbounded();

        Self {
            sim_read_chan: src,
            sim_write_chan: ssc,
            rend_read_chan: rrc,
            rend_write_chan: rsc,
            gui_read_chan: grc,
            gui_write_chan: gsc,
        }
    }
}
