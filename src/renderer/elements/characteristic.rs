use rkyv::{Archive, Deserialize, Serialize};
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize, EnumIter)]
#[strum(serialize_all = "title_case")]
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
            Self::Transitive => write!(f, "transitive"),
            Self::FunctionalProperty => write!(f, "functional"),
            Self::InverseFunctionalProperty => write!(f, "inverse functional"),
            Self::ReflexiveProperty => write!(f, "reflexive"),
            Self::IrreflexiveProperty => write!(f, "irreflexive"),
            Self::SymmetricProperty => write!(f, "symmetric"),
            Self::AsymmetricProperty => write!(f, "asymmetric"),
            Self::HasKey => write!(f, "key"),
        }
    }
}
