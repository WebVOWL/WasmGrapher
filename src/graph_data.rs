pub use crate::renderer::elements::{
    characteristic::Characteristic, element_type::ElementType, owl::*, rdf::*, rdfs::*,
};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Struct containing graph data for grapher
#[repr(C)]
#[derive(Archive, Deserialize, Serialize, PartialEq, Eq, Clone, Default)]
pub struct GraphDisplayData {
    /// Labels annotate classes and properties
    ///
    /// The index into this vector is the ID of the node/edge having a label.
    /// The ID is defined by the indices of `elements`.
    pub labels: Vec<Option<String>>,
    /// Elements are the nodes and edge types for which visualization is supported.
    ///
    /// The index into this vector determines the unique ID of each element.
    pub elements: Vec<ElementType>,
    /// An array of three elements: `source node`, `edge`, and `target node`.
    ///
    /// The elements of the array are node/edge IDs.
    /// They are defined by the indices of `elements`.
    pub edges: Vec<[usize; 3]>,
    /// Cardinalities of edges.
    ///
    /// The tuple consists of 2 elements:
    ///     - u32: The index of the edge in `edges`.
    ///     - (String, Option<String>):
    ///         - String: The min cardinality of the edge.
    ///         - Option<String>: The max cardinality of the target edge.
    pub cardinalities: Vec<(u32, (String, Option<String>))>,
    /// Special node types. For instance "transitive" or "inverse functional".
    ///
    /// The hashmap consists of:
    ///     - usize: The ID of the node. Defined by the indices of `elements`.
    ///     - String: The name of the node type. E.g. "transitive".
    pub characteristics: HashMap<usize, HashSet<Characteristic>>,
    /// Track number of individuals connected to a specific element
    pub individual_counts: HashMap<usize, u32>,
}

impl GraphDisplayData {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn demo() -> Self {
        let labels = vec![
            Some(String::from("My class")),
            Some(String::from("Rdfs class")),
            Some(String::from("Rdfs resource")),
            Some(String::from("Loooooooong class 1 2 3 4 5 6 7 8 9")),
            Some(String::from("Thing")),
            Some(String::from("Eq1\nEq2\nEq3")),
            Some(String::from("Deprecated")),
            None,
            Some(String::from("Literal")),
            None,
            Some(String::from("DisjointUnion 1 2 3 4 5 6 7 8 9")),
            None,
            None,
            Some(String::from("This Datatype is very long")),
            Some(String::from("AllValues")),
            Some(String::from("Property1")),
            Some(String::from("Property2")),
            None,
            None,
            Some(String::from("is a")),
            Some(String::from("Deprecated")),
            Some(String::from("External")),
            Some(String::from("Symmetric")),
            Some(String::from("Property\nInverseProperty")),
            None,
            None,
        ];
        let elements = vec![
            ElementType::Owl(OwlType::Node(OwlNode::Class)),
            ElementType::Rdfs(RdfsType::Node(RdfsNode::Class)),
            ElementType::Rdfs(RdfsType::Node(RdfsNode::Resource)),
            ElementType::Owl(OwlType::Node(OwlNode::ExternalClass)),
            ElementType::Owl(OwlType::Node(OwlNode::Thing)),
            ElementType::Owl(OwlType::Node(OwlNode::EquivalentClass)),
            ElementType::Owl(OwlType::Node(OwlNode::DeprecatedClass)),
            ElementType::Owl(OwlType::Node(OwlNode::AnonymousClass)),
            ElementType::Rdfs(RdfsType::Node(RdfsNode::Literal)),
            ElementType::Owl(OwlType::Node(OwlNode::Complement)),
            ElementType::Owl(OwlType::Node(OwlNode::DisjointUnion)),
            ElementType::Owl(OwlType::Node(OwlNode::IntersectionOf)),
            ElementType::Owl(OwlType::Node(OwlNode::UnionOf)),
            ElementType::Rdfs(RdfsType::Node(RdfsNode::Datatype)),
            ElementType::Owl(OwlType::Edge(OwlEdge::ValuesFrom)),
            ElementType::Owl(OwlType::Edge(OwlEdge::DatatypeProperty)),
            ElementType::Owl(OwlType::Edge(OwlEdge::DatatypeProperty)),
            ElementType::Rdfs(RdfsType::Edge(RdfsEdge::SubclassOf)),
            ElementType::Owl(OwlType::Edge(OwlEdge::DisjointWith)),
            ElementType::Rdf(RdfType::Edge(RdfEdge::RdfProperty)),
            ElementType::Owl(OwlType::Edge(OwlEdge::DeprecatedProperty)),
            ElementType::Owl(OwlType::Edge(OwlEdge::ExternalProperty)),
            ElementType::Owl(OwlType::Edge(OwlEdge::ObjectProperty)),
            ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf)),
            ElementType::NoDraw,
            ElementType::NoDraw,
        ];
        let edges = vec![
            [0, 14, 1],
            [13, 15, 8],
            [8, 16, 13],
            [0, 17, 3],
            [9, 18, 12],
            [1, 19, 2],
            [10, 24, 11],
            [11, 25, 12],
            [6, 20, 7],
            [6, 21, 7],
            [4, 22, 4],
            [2, 23, 5],
            [5, 23, 2],
        ];
        let cardinalities: Vec<(u32, (String, Option<String>))> = vec![
            (0, ("∀".to_string(), None)),
            (8, ("1".to_string(), None)),
            (1, ("1".to_string(), Some("10".to_string()))),
            (10, ("5".to_string(), Some("10".to_string()))),
        ];
        let mut characteristics = HashMap::new();
        characteristics.insert(21, HashSet::from_iter([Characteristic::TransitiveProperty]));
        characteristics.insert(
            23,
            HashSet::from_iter([
                Characteristic::FunctionalProperty,
                Characteristic::InverseFunctionalProperty,
            ]),
        );

        let mut individual_counts = HashMap::new();
        individual_counts.insert(0, 2);

        Self {
            labels,
            elements,
            edges,
            cardinalities,
            characteristics,
            individual_counts,
        }
    }
}

impl std::fmt::Display for GraphDisplayData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GraphDisplayData {{")?;
        writeln!(
            f,
            "\tlabels: {:#?}",
            self.labels
                .iter()
                .enumerate()
                .map(|(i, label)| format!("[{i}] {}", label.as_ref().map_or("(None)", |v| v)))
                .collect::<Vec<_>>()
        )?;
        writeln!(
            f,
            "\telements: {:#?}",
            self.elements
                .iter()
                .enumerate()
                .map(|(i, element)| format!("[{i}] {element}"))
                .collect::<Vec<_>>()
        )?;
        writeln!(
            f,
            "\tedges: {:#?}",
            self.edges
                .iter()
                .map(|edge| format!("{edge:?}"))
                .collect::<Vec<_>>()
        )?;
        writeln!(f, "\tcardinalities: {:?}", self.cardinalities)?;
        writeln!(f, "\tcharacteristics: {:?}", self.characteristics)?;
        writeln!(f, "\tindividual_counts: {:?}", self.individual_counts)?;
        writeln!(f, "}}")
    }
}

#[cfg(feature = "test-utils")]
mod test_utils {

    use super::{ElementType, GraphDisplayData, OwlEdge, OwlType};
    use log::warn;
    use sovs_parser::{Properties, SovsError, Specification, SpecificationBuilder};

    impl TryFrom<GraphDisplayData> for Specification {
        type Error = sovs_parser::SovsError;

        fn try_from(value: GraphDisplayData) -> Result<Self, SovsError> {
            let mut builder = SpecificationBuilder::new();

            for (i, (label, element)) in value.labels.iter().zip(&value.elements).enumerate() {
                if element.is_node() {
                    builder.node(i.to_string(), node_properties(label.clone(), *element));
                    continue;
                }

                #[expect(clippy::expect_used)]
                let [from, _, to] = value
                    .edges
                    .iter()
                    .find(|[_, edge, _]| *edge == i)
                    .expect("edge does not exist in edge array");

                if *element == ElementType::Owl(OwlType::Edge(OwlEdge::InverseOf)) {
                    let Some((inverse_label, main_label)) =
                        label.as_ref().and_then(|l| l.split_once('\n'))
                    else {
                        warn!("invalid InverseOf edge: Either no newline or no label: {label:?}");
                        continue;
                    };
                    let mut main_props = Properties::new();
                    main_props.insert("text".to_owned(), main_label.to_owned());
                    main_props.insert("kind".to_owned(), "owl:objectProperty".to_owned());

                    if let Some(characteristics) = value.characteristics.get(&i) {
                        for c in characteristics {
                            main_props
                                .insert("characteristics".to_string(), c.as_sovs().to_owned());
                        }
                    }

                    builder.edge(i.to_string(), from.to_string(), to.to_string(), main_props);

                    let mut inverse_props = Properties::new();

                    inverse_props.insert("text".to_owned(), inverse_label.to_owned());
                    inverse_props.insert("kind".to_owned(), "owl:inverseProperty".to_owned());

                    if let Some(characteristics) = value.characteristics.get(&i) {
                        for c in characteristics {
                            inverse_props
                                .insert("characteristics".to_string(), c.as_sovs().to_owned());
                        }
                    }

                    inverse_props.insert("inverseOf".to_owned(), i.to_string());

                    builder.edge(
                        format!("{i}_inv"),
                        to.to_string(),
                        from.to_string(),
                        inverse_props,
                    );

                    continue;
                }

                builder.edge(
                    i.to_string(),
                    from.to_string(),
                    to.to_string(),
                    edge_properties(&value, i),
                );
            }

            builder.build()
        }
    }

    fn node_properties(label: Option<String>, element_type: ElementType) -> Properties {
        let mut properties = Properties::new();
        if let Some(label) = label {
            properties.insert("text".to_string(), label);
        }
        if let Some(kind) = element_type.sovs_kind() {
            let kind = kind.to_string();
            properties.insert("kind".to_string(), kind);
        }
        properties
    }

    fn edge_properties(data: &GraphDisplayData, index: usize) -> Properties {
        let label = &data.labels[index];
        let element_type = data.elements[index];
        let mut properties = Properties::new();
        if let Some(label) = label {
            properties.insert("text".to_string(), label.clone());
        }
        if let Some(kind) = element_type.sovs_kind() {
            let kind = kind.to_string();
            properties.insert("kind".to_string(), kind);
        }

        if let Some(characteristics) = data.characteristics.get(&index) {
            for c in characteristics {
                properties.insert("characteristics".to_string(), c.as_sovs().to_owned());
            }
        }

        properties
    }
}
