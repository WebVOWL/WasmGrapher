use rkyv::{Archive, Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum Characteristic {
    Transitive,
    FunctionalProperty,
    InverseFunctionalProperty,
    ReflexiveProperty,
    IrreflexiveProperty,
    SymmetricProperty,
    AsymmetricProperty,
    HasKey,
}

impl std::fmt::Display for Characteristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Characteristic::Transitive => write!(f, "transitive"),
            Characteristic::FunctionalProperty => write!(f, "functional"),
            Characteristic::InverseFunctionalProperty => write!(f, "inverse functional"),
            Characteristic::ReflexiveProperty => write!(f, "reflexive"),
            Characteristic::IrreflexiveProperty => write!(f, "irreflexive"),
            Characteristic::SymmetricProperty => write!(f, "symmetric"),
            Characteristic::AsymmetricProperty => write!(f, "asymmetric"),
            Characteristic::HasKey => write!(f, "key"),
        }
    }
}
