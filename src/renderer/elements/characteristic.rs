use rkyv::{Archive, Deserialize, Serialize};
use strum::EnumIter;

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
    strum::AsRefStr,
)]
#[strum(serialize_all = "title_case")]
#[rkyv(derive(Hash, Eq, PartialEq))]
pub enum Characteristic {
    #[strum(serialize = "transitive")]
    TransitiveProperty,
    #[strum(serialize = "functional")]
    FunctionalProperty,
    #[strum(serialize = "inverse functional")]
    InverseFunctionalProperty,
    #[strum(serialize = "reflexive")]
    ReflexiveProperty,
    #[strum(serialize = "irreflexive")]
    IrreflexiveProperty,
    #[strum(serialize = "symmetric")]
    SymmetricProperty,
    #[strum(serialize = "asymmetric")]
    AsymmetricProperty,
    #[strum(serialize = "has key")]
    HasKey,
}

impl Characteristic {
    #[cfg(feature = "test-utils")]
    #[must_use]
    pub const fn as_sovs(self) -> &'static str {
        match self {
            Self::AsymmetricProperty => "owl:asymmetricProperty",
            Self::FunctionalProperty => "owl:functionalProperty",
            Self::InverseFunctionalProperty => "owl:inverseFunctionalProperty",
            Self::IrreflexiveProperty => "owl:irreflexiveProperty",
            Self::ReflexiveProperty => "owl:reflexiveProperty",
            Self::SymmetricProperty => "owl:symmetricProperty",
            Self::TransitiveProperty => "owl:transitiveProperty",
            Self::HasKey => "owl:hasKey",
        }
    }
}
