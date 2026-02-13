use super::SparqlSnippet;
use rkyv::{Archive, Deserialize, Serialize};
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

impl SparqlSnippet for GenericNode {
    fn snippet(self) -> &'static str {
        match self {
            GenericNode::Generic => todo!(),
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

impl SparqlSnippet for GenericEdge {
    fn snippet(self) -> &'static str {
        match self {
            GenericEdge::Generic => todo!(),
        }
    }
}
