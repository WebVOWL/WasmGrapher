pub mod characteristic;
pub mod element_type;
pub mod generic;
pub mod owl;
pub mod rdf;
pub mod rdfs;

pub trait SparqlSnippet {
    /// Get the SPARQL snippet representing `self`.
    fn snippet(self) -> &'static str;
}
