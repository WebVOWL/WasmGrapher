use crate::prelude::{OwlEdge, OwlNode};

use super::{generic::GenericType, owl::OwlType, rdf::RdfType, rdfs::RdfsType, xsd::XSDType};
use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use std::num::TryFromIntError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum ElementType {
    Owl(OwlType),
    Rdf(RdfType),
    Rdfs(RdfsType),
    Xsd(XSDType),
    Generic(GenericType),
    NoDraw,
}

impl ElementType {
    #[must_use]
    pub const fn is_edge(self) -> bool {
        match self {
            Self::Owl(OwlType::Edge(_))
            | Self::Rdf(RdfType::Edge(_))
            | Self::Rdfs(RdfsType::Edge(_))
            | Self::Generic(GenericType::Edge(_)) => true,
            Self::Owl(OwlType::Node(_))
            | Self::Rdf(RdfType::Node(_))
            | Self::Rdfs(RdfsType::Node(_))
            | Self::Generic(GenericType::Node(_))
            | Self::Xsd(XSDType::Node(_))
            | Self::NoDraw => false,
        }
    }

    #[must_use]
    pub const fn is_node(self) -> bool {
        !self.is_edge()
    }

    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> Option<&'static str> {
        match self {
            Self::Owl(owl) => Some(owl.sovs_kind()),
            Self::Rdf(rdf) => Some(rdf.sovs_kind()),
            Self::Rdfs(rdfs) => Some(rdfs.sovs_kind()),
            Self::Generic(generic) => generic.sovs_kind(),
            Self::Xsd(xsd) => Some(xsd.sovs_kind()),
            Self::NoDraw => None,
        }
    }
}

impl Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDraw => write!(f, "NoDraw"),
            Self::Rdf(RdfType::Node(node)) => node.fmt(f),
            Self::Rdf(RdfType::Edge(edge)) => edge.fmt(f),
            Self::Rdfs(RdfsType::Node(node)) => node.fmt(f),
            Self::Rdfs(RdfsType::Edge(edge)) => edge.fmt(f),
            Self::Owl(OwlType::Node(node)) => node.fmt(f),
            Self::Owl(OwlType::Edge(edge)) => edge.fmt(f),
            Self::Generic(GenericType::Node(node)) => node.fmt(f),
            Self::Generic(GenericType::Edge(edge)) => edge.fmt(f),
            Self::Xsd(XSDType::Node(node)) => node.fmt(f),
        }
    }
}

impl From<ElementType> for i128 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        Self::from(out)
    }
}

impl From<ElementType> for i64 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        Self::from(out)
    }
}

impl TryFrom<ElementType> for i32 {
    type Error = TryFromIntError;

    fn try_from(value: ElementType) -> Result<Self, Self::Error> {
        let out: u32 = value.into();
        Self::try_from(out)
    }
}

impl From<ElementType> for u128 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        Self::from(out)
    }
}

impl From<ElementType> for u64 {
    fn from(value: ElementType) -> Self {
        let out: u32 = value.into();
        Self::from(out)
    }
}

impl From<ElementType> for u32 {
    #[doc =  include_str!("../../../Visualization.md")]
    fn from(value: ElementType) -> Self {
        match value {
            ElementType::NoDraw => 0,
            ElementType::Rdf(RdfType::Node(node)) => node.into(),
            ElementType::Rdf(RdfType::Edge(edge)) => edge.into(),
            ElementType::Rdfs(RdfsType::Node(node)) => node.into(),
            ElementType::Rdfs(RdfsType::Edge(edge)) => edge.into(),
            ElementType::Owl(OwlType::Node(node)) => node.into(),
            ElementType::Owl(OwlType::Edge(edge)) => edge.into(),
            ElementType::Generic(GenericType::Node(node)) => node.into(),
            ElementType::Generic(GenericType::Edge(edge)) => edge.into(),
            ElementType::Xsd(XSDType::Node(node)) => node.into(),
        }
    }
}
