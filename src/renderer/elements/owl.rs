use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::{Display, write};
use strum::EnumIter;

// TODO: Expand with OWL 2

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum OwlType {
    Node(OwlNode),
    Edge(OwlEdge),
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
pub enum OwlNode {
    AnonymousClass,
    Class,
    Complement,
    DeprecatedClass,
    ExternalClass,
    EquivalentClass,
    DisjointUnion,
    IntersectionOf,
    Thing,
    UnionOf,
}

impl From<OwlNode> for u32 {
    fn from(value: OwlNode) -> Self {
        match value {
            OwlNode::AnonymousClass => 30000,
            OwlNode::Class => 30001,
            OwlNode::Complement => 30002,
            OwlNode::DeprecatedClass => 30003,
            OwlNode::ExternalClass => 30004,
            OwlNode::EquivalentClass => 30005,
            OwlNode::DisjointUnion => 30006,
            OwlNode::IntersectionOf => 30007,
            OwlNode::Thing => 30008,
            OwlNode::UnionOf => 30009,
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
pub enum OwlEdge {
    DatatypeProperty,
    DisjointWith,
    DeprecatedProperty,
    ExternalProperty,
    InverseOf,
    ObjectProperty,
    ValuesFrom,
}

impl From<OwlEdge> for u32 {
    fn from(value: OwlEdge) -> Self {
        match value {
            OwlEdge::DatatypeProperty => 35000,
            OwlEdge::DisjointWith => 35001,
            OwlEdge::DeprecatedProperty => 35002,
            OwlEdge::ExternalProperty => 35003,
            OwlEdge::InverseOf => 35004,
            OwlEdge::ObjectProperty => 35005,
            OwlEdge::ValuesFrom => 35006,
        }
    }
}
