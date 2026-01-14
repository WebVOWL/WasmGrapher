use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfsType {
    Node(RdfsNode),
    Edge(RdfsEdge),
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
pub enum RdfsNode {
    #[strum(serialize = "RDFS Class")]
    Class,
    Literal,
    #[strum(serialize = "RDFS Resource")]
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
