use super::SparqlSnippet;
use rkyv::{Archive, Deserialize, Serialize};
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum RdfType {
    // Node(RdfNode),
    Edge(RdfEdge),
}

// #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
// pub enum RdfNode {}

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

impl From<RdfEdge> for u32 {
    fn from(value: RdfEdge) -> Self {
        match value {
            RdfEdge::RdfProperty => 15000,
        }
    }
}

impl SparqlSnippet for RdfEdge {
    fn snippet(self) -> &'static str {
        match self {
            RdfEdge::RdfProperty => {
                r#"{
                ?id rdf:Property ?target
                BIND(rdf:Property AS ?nodeType)
                }"#
            }
        }
    }
}
