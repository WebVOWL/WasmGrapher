use super::{generic::GenericType, owl::OwlType, rdf::RdfType, rdfs::RdfsType};
use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use std::num::TryFromIntError;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum ElementType {
    Owl(OwlType),
    Rdf(RdfType),
    Rdfs(RdfsType),
    Generic(GenericType),
    NoDraw,
}

impl Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoDraw => write!(f, "NoDraw"),
            // Self::Rdf(RdfType::Node(node)) => match node {},
            Self::Rdf(RdfType::Edge(edge)) => edge.fmt(f),
            Self::Rdfs(RdfsType::Node(node)) => node.fmt(f),
            Self::Rdfs(RdfsType::Edge(edge)) => edge.fmt(f),
            Self::Owl(OwlType::Node(node)) => node.fmt(f),
            Self::Owl(OwlType::Edge(edge)) => edge.fmt(f),
            Self::Generic(GenericType::Node(node)) => node.fmt(f),
            Self::Generic(GenericType::Edge(edge)) => edge.fmt(f),
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
            // ElementType::Rdf(RdfType::Node(node)) => match node {},
            ElementType::Rdf(RdfType::Edge(edge)) => edge.into(),
            ElementType::Rdfs(RdfsType::Node(node)) => node.into(),
            ElementType::Rdfs(RdfsType::Edge(edge)) => edge.into(),
            ElementType::Owl(OwlType::Node(node)) => node.into(),
            ElementType::Owl(OwlType::Edge(edge)) => edge.into(),
            ElementType::Generic(GenericType::Node(node)) => node.into(),
            ElementType::Generic(GenericType::Edge(edge)) => edge.into(),
        }
    }
}
