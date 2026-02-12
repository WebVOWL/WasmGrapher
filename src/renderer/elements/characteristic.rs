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
    Transitive,
    FunctionalProperty,
    InverseFunctionalProperty,
    ReflexiveProperty,
    IrreflexiveProperty,
    SymmetricProperty,
    AsymmetricProperty,
    HasKey,
}
