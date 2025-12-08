pub enum Characteristic {
    Transitive,
    FunctionalProperty,
    InverseFunctionalProperty,
    Reflexive,
    Irreflexive,
    Symmetric,
    Asymmetric,
    HasKey,
}

impl std::fmt::Display for Characteristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Characteristic::Transitive => write!(f, "transitive"),
            Characteristic::FunctionalProperty => write!(f, "functional"),
            Characteristic::InverseFunctionalProperty => write!(f, "inverse functional"),
            Characteristic::Reflexive => write!(f, "reflexive"),
            Characteristic::Irreflexive => write!(f, "irreflexive"),
            Characteristic::Symmetric => write!(f, "symmetric"),
            Characteristic::Asymmetric => write!(f, "asymmetric"),
            Characteristic::HasKey => write!(f, "key"),
        }
    }
}
