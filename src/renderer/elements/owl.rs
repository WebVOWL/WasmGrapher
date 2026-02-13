use super::SparqlSnippet;
use rkyv::{Archive, Deserialize, Serialize};
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

impl SparqlSnippet for OwlNode {
    fn snippet(self) -> &'static str {
        match self {
            OwlNode::AnonymousClass => {
                r#"{
                ?id a owl:Class
                FILTER(!isIRI(?id))
                BIND("blanknode" AS ?nodeType)
                }"#
            }
            OwlNode::Class => {
                r#"{
                ?id a owl:Class .
                FILTER(isIRI(?id))
                BIND(owl:Class AS ?nodeType)
                }"#
            }
            OwlNode::Complement => {
                r#"{
                ?id owl:complementOf ?target .
                BIND(owl:complementOf AS ?nodeType)
                }"#
            }
            OwlNode::DeprecatedClass => {
                r#"{
                ?id a owl:DeprecatedClass .
                BIND(owl:DeprecatedClass AS ?nodeType)
                }"#
            }
            OwlNode::ExternalClass => {
                // Not handled here as externals uses identical
                // logic across classes and properties.
                ""
            }
            OwlNode::EquivalentClass => {
                r#"{
                ?id owl:equivalentClass ?target
                BIND(owl:equivalentClass AS ?nodeType)
                }"#
            }
            OwlNode::DisjointUnion => {
                r#"{
                ?id owl:disjointUnionOf ?target .
                BIND(owl:disjointUnionOf AS ?nodeType)
                }"#
            }
            OwlNode::IntersectionOf => {
                r#"{
                ?id owl:intersectionOf ?target .
                BIND(owl:intersectionOf AS ?nodeType)
                }"#
            }
            OwlNode::Thing => {
                r#"{
                ?id a owl:Thing .
                BIND(owl:Thing AS ?nodeType)
                }"#
            }
            OwlNode::UnionOf => {
                r#"{
                ?id owl:unionOf ?list .
                BIND(owl:unionOf AS ?nodeType)
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

impl SparqlSnippet for OwlEdge {
    fn snippet(self) -> &'static str {
        match self {
            OwlEdge::DatatypeProperty => {
                r#"{
                ?id owl:DatatypeProperty ?target
                BIND(owl:DatatypeProperty AS ?nodeType)
                }"#
            }
            OwlEdge::DisjointWith => {
                r#"{
                ?id owl:disjointWith ?target
                BIND(owl:disjointWith AS ?nodeType)
                }"#
            }
            OwlEdge::DeprecatedProperty => {
                r#"{
                ?id a owl:DeprecatedProperty .
                BIND(owl:DeprecatedProperty AS ?nodeType)
                }"#
            }
            OwlEdge::ExternalProperty => {
                // Not handled here as externals uses identical
                // logic across classes and properties.
                ""
            }
            OwlEdge::InverseOf => {
                r#"{
                ?id owl:inverseOf ?target .
                BIND(owl:inverseOf AS ?nodeType)
                }"#
            }
            OwlEdge::ObjectProperty => {
                r#"{
                ?id a owl:ObjectProperty
                BIND(owl:ObjectProperty AS ?nodeType)
                }"#
            }
            OwlEdge::ValuesFrom => {
                r#"{
                {
                    ?id owl:someValuesFrom ?target .
                }
                UNION
                {
                    ?id owl:allValuesFrom ?target .
                }
                BIND("ValuesFrom" AS ?nodeType)
                }"#
            }
        }
    }
}
