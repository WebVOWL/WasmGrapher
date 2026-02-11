use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::Display;

#[derive(Archive, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Cardinality {
    AllValuesFrom,
    SomeValuesFrom,
    HasSelf,
    HasValue,
    Min(u64),
    Max(u64),
    Exact(u64),
    Range(u64, u64),
}

impl Display for Cardinality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AllValuesFrom => todo!(),
            Self::SomeValuesFrom => todo!(),
            Self::HasSelf => todo!(),
            Self::HasValue => todo!(),
            Self::Min(x) => write!(f, "{x}..*"),
            Self::Max(x) => write!(f, "0..{x}"),
            Self::Exact(x) => write!(f, "{x}"),
            Self::Range(min, max) => write!(f, "{min}..{max}"),
        }
    }
}
