use crate::{prelude::SimulatorEvent, renderer::events::RenderEvent};
use flume::{Receiver, Sender};

#[expect(clippy::struct_field_names)]
pub struct EventDispatcher {
    /// Receiver must only be consumed by the simulator
    pub sim_read_chan: Receiver<SimulatorEvent>,
    pub sim_write_chan: Sender<SimulatorEvent>,
    /// Receiver must only be consumed by the renderer
    pub rend_read_chan: Receiver<RenderEvent>,
    pub rend_write_chan: Sender<RenderEvent>,
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

        Self {
            sim_read_chan: src,
            sim_write_chan: ssc,
            rend_read_chan: rrc,
            rend_write_chan: rsc,
        }
    }
}
