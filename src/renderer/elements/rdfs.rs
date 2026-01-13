use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfsType {
    Node(RdfsNode),
    Edge(RdfsEdge),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize, EnumIter)]
pub enum RdfsNode {
    Class,
    Literal,
    Resource,
    Datatype,
}

impl From<RdfsNode> for u32 {
    fn from(value: RdfsNode) -> Self {
        match value {
            RdfsNode::Class => 20000,
            RdfsNode::Literal => 20001,
            RdfsNode::Resource => 20002,
            RdfsNode::Datatype => 20003,
        }
    }
}

impl Display for RdfsNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfsNode::Class => write!(f, "RDFS Class"),
            RdfsNode::Literal => write!(f, "Literal"),
            RdfsNode::Resource => write!(f, "Resource"),
            RdfsNode::Datatype => write!(f, "Datatype"),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize, EnumIter)]
pub enum RdfsEdge {
    SubclassOf,
}

impl From<RdfsEdge> for u32 {
    fn from(value: RdfsEdge) -> Self {
        match value {
            RdfsEdge::SubclassOf => 25000,
        }
    }
}

impl Display for RdfsEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RdfsEdge::SubclassOf => write!(f, "Subclass Of"),
        }
    }
}
