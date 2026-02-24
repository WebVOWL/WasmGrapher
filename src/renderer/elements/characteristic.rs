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
)]
#[strum(serialize_all = "title_case")]
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
