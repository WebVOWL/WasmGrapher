use super::SparqlSnippet;
use rkyv::{Archive, Deserialize, Serialize};
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

impl SparqlSnippet for RdfsNode {
    fn snippet(self) -> &'static str {
        match self {
            RdfsNode::Class => {
                r#"{
                ?id a rdfs:Class .
                FILTER(?id != owl:Class)
                BIND(rdfs:Class AS ?nodeType)
                }"#
            }
            RdfsNode::Literal => {
                r#"{
                ?id a rdfs:Literal .
                BIND(rdfs:Literal AS ?nodeType)
                }"#
            }
            RdfsNode::Resource => {
                r#"{
                ?id a rdfs:Resource .
                FILTER(isIRI(?id) || isBlank(?id))
                BIND(rdfs:Resource AS ?nodeType)
                }"#
            }
            RdfsNode::Datatype => {
                r#"{
                ?id rdfs:Datatype ?target
                BIND(rdfs:Datatype AS ?nodeType)
                }"#
            }
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

impl SparqlSnippet for RdfsEdge {
    fn snippet(self) -> &'static str {
        match self {
            RdfsEdge::SubclassOf => {
                r#"{
                ?id rdfs:subClassOf ?target
                BIND(rdfs:subClassOf AS ?nodeType)
                }"#
            }
        }
    }
}
