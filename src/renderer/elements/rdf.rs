use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfType {
    Node(RdfNode),
    Edge(RdfEdge),
}

impl RdfType {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::Node(node) => node.sovs_kind(),
            Self::Edge(edge) => edge.sovs_kind(),
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
pub enum RdfNode {
    HTML,
    PlainLiteral,
    XMLLiteral,
}

impl RdfNode {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::HTML => "rdf:html",
            Self::PlainLiteral => "rdf:plainLiteral",
            Self::XMLLiteral => "rdf:xMLLiteral",
        }
    }
}

impl From<RdfNode> for u32 {
    fn from(value: RdfNode) -> Self {
        match value {
            RdfNode::HTML => 10000,
            RdfNode::PlainLiteral => 10001,
            RdfNode::XMLLiteral => 10002,
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
pub enum RdfEdge {
    #[strum(serialize = "is a")]
    RdfProperty,
}

impl RdfEdge {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::RdfProperty => "rdf:property",
        }
    }
}

impl From<RdfEdge> for u32 {
    fn from(value: RdfEdge) -> Self {
        match value {
            RdfEdge::RdfProperty => 15000,
        }
    }
}
