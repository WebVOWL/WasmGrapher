use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::{Display, write};
use strum::EnumIter;

// TODO: Expand with OWL 2

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum OwlType {
    Node(OwlNode),
    Edge(OwlEdge),
}

impl OwlType {
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

impl OwlNode {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::AnonymousClass => "anonymous",
            Self::Class => "owl:class",
            Self::Complement => "owl:complementOf",
            Self::DeprecatedClass => "owl:deprecatedClass",
            Self::ExternalClass => "external",
            Self::EquivalentClass => "owl:equivalentClass",
            Self::DisjointUnion => "owl:disjointUnion",
            Self::IntersectionOf => "owl:intersectionOf",
            Self::Thing => "owl:thing",
            Self::UnionOf => "owl:unionOf",
        }
    }
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

impl OwlEdge {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::DatatypeProperty => "owl:datatypeProperty",
            Self::DisjointWith => "owl:disjointWith",
            Self::DeprecatedProperty => "owl:deprecatedProperty",
            Self::ExternalProperty => "external",
            Self::InverseOf => "owl:inverseOf",
            Self::ObjectProperty => "owl:objectProperty",
            Self::ValuesFrom => "owl:valuesFrom",
        }
    }
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
