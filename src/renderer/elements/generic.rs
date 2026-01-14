use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum GenericType {
    Node(GenericNode),
    Edge(GenericEdge),
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Archive,
    Deserialize,
    Serialize,
    EnumIter,
    strum::Display,
)]
#[strum(serialize_all = "title_case")]
pub enum GenericNode {
    #[strum(serialize = "Generic Node")]
    Generic,
}

impl From<GenericNode> for u32 {
    fn from(value: GenericNode) -> Self {
        match value {
            GenericNode::Generic => 40000,
        }
    }
}

#[derive(
    Copy,
    Clone,
    Debug,
    PartialEq,
    Eq,
    Hash,
    Archive,
    Deserialize,
    Serialize,
    EnumIter,
    strum::Display,
)]
#[strum(serialize_all = "title_case")]
pub enum GenericEdge {
    #[strum(serialize = "Generic Edge")]
    Generic,
}

impl From<GenericEdge> for u32 {
    fn from(value: GenericEdge) -> Self {
        match value {
            GenericEdge::Generic => 50000,
        }
    }
}
