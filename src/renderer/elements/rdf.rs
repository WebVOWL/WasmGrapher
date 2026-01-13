use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfType {
    // Node(RdfNode),
    Edge(RdfEdge),
}

// #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
// pub enum RdfNode {}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize, EnumIter)]
pub enum RdfEdge {
    RdfProperty,
}

impl From<RdfEdge> for u32 {
    fn from(value: RdfEdge) -> Self {
        match value {
            RdfEdge::RdfProperty => 15000,
        }
    }
}

impl Display for RdfEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfEdge::RdfProperty => write!(f, "is a"),
        }
    }
}
