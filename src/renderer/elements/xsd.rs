use rkyv::{Archive, Deserialize, Serialize};
use std::fmt::{Display, write};
use strum::EnumIter;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Archive, Deserialize, Serialize)]
pub enum XSDType {
    Node(XSDNode),
    // Edge(XSDEdge),
}

impl XSDType {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::Node(node) => node.sovs_kind(),
            // Self::Edge(edge) => edge.sovs_kind(),
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
pub enum XSDNode {
    Int,
    Integer,
    NegativeInteger,
    NonNegativeInteger,
    NonPositiveInteger,
    PositiveInteger,
    UnsignedInt,
    UnsignedLong,
    UnsignedShort,
    Decimal,
    Float,
    Double,
    Short,
    Long,
    Date,
    DataTime,
    DateTimeStamp,
    Duration,
    GDay,
    GMonth,
    GMonthDay,
    GYear,
    GYearMonth,
    Time,
    AnyURI,
    ID,
    #[strum(serialize = "IDREF")]
    Idref,
    Language,
    #[strum(serialize = "NMTOKEN")]
    Nmtoken,
    Name,
    NCName,
    QName,
    String,
    Token,
    NormalizedString,
    #[strum(serialize = "NOTATION")]
    Notation,
    AnySimpleType,
    Base64Binary,
    Boolean,
    #[strum(serialize = "ENTITY")]
    Entity,
    UnsignedByte,
    Byte,
    HexBinary,
}

impl XSDNode {
    #[cfg(feature = "test-utils")]
    pub(crate) const fn sovs_kind(self) -> &'static str {
        match self {
            Self::Int => "xsd:int",
            Self::Integer => "xsd:integer",
            Self::NegativeInteger => "xsd:negativeInteger",
            Self::NonNegativeInteger => "xsd:nonNegativeInteger",
            Self::NonPositiveInteger => "xsd:nonPositiveInteger",
            Self::PositiveInteger => "xsd:positiveInteger",
            Self::UnsignedInt => "xsd:unsignedInt",
            Self::UnsignedLong => "xsd:unsignedLong",
            Self::UnsignedShort => "xsd:unsignedShort",
            Self::Decimal => "xsd:decimal",
            Self::Float => "xsd:float",
            Self::Double => "xsd:double",
            Self::Short => "xsd:short",
            Self::Long => "xsd:long",
            Self::Date => "xsd:date",
            Self::DataTime => "xsd:dateTime",
            Self::DateTimeStamp => "xsd:dateTimeStamp",
            Self::Duration => "xsd:duration",
            Self::GDay => "xsd:gDay",
            Self::GMonth => "xsd:gMonth",
            Self::GMonthDay => "xsd:gMonthDay",
            Self::GYear => "xsd:gYear",
            Self::GYearMonth => "xsd:gYearMonth",
            Self::Time => "xsd:time",
            Self::AnyURI => "xsd:anyURI",
            Self::ID => "xsd:id",
            Self::Idref => "xsd:idref",
            Self::Language => "xsd:language",
            Self::Nmtoken => "xsd:nmtoken",
            Self::Name => "xsd:name",
            Self::NCName => "xsd:ncName",
            Self::QName => "xsd:qName",
            Self::String => "xsd:string",
            Self::Token => "xsd:token",
            Self::NormalizedString => "xsd:normalizedString",
            Self::Notation => "xsd:notation",
            Self::AnySimpleType => "xsd:anySimpleType",
            Self::Base64Binary => "xsd:base64Binary",
            Self::Boolean => "xsd:boolean",
            Self::Entity => "xsd:entity",
            Self::UnsignedByte => "xsd:unsignedByte",
            Self::Byte => "xsd:byte",
            Self::HexBinary => "xsd:hexBinary",
        }
    }
}

impl From<XSDNode> for u32 {
    fn from(value: XSDNode) -> Self {
        match value {
            XSDNode::Int => 60001,
            XSDNode::Integer => 60002,
            XSDNode::NegativeInteger => 60003,
            XSDNode::NonNegativeInteger => 60004,
            XSDNode::NonPositiveInteger => 60005,
            XSDNode::PositiveInteger => 60006,
            XSDNode::UnsignedInt => 60007,
            XSDNode::UnsignedLong => 60008,
            XSDNode::UnsignedShort => 60009,
            XSDNode::Decimal => 60010,
            XSDNode::Float => 60011,
            XSDNode::Double => 60012,
            XSDNode::Short => 60013,
            XSDNode::Long => 60014,
            XSDNode::Date => 60015,
            XSDNode::DataTime => 60016,
            XSDNode::DateTimeStamp => 60017,
            XSDNode::Duration => 60018,
            XSDNode::GDay => 60019,
            XSDNode::GMonth => 60020,
            XSDNode::GMonthDay => 60021,
            XSDNode::GYear => 60022,
            XSDNode::GYearMonth => 60023,
            XSDNode::Time => 60024,
            XSDNode::AnyURI => 60025,
            XSDNode::ID => 60026,
            XSDNode::Idref => 60027,
            XSDNode::Language => 60028,
            XSDNode::Nmtoken => 60029,
            XSDNode::Name => 60030,
            XSDNode::NCName => 60031,
            XSDNode::QName => 60032,
            XSDNode::String => 60033,
            XSDNode::Token => 60034,
            XSDNode::NormalizedString => 60035,
            XSDNode::Notation => 60036,
            XSDNode::AnySimpleType => 60037,
            XSDNode::Base64Binary => 60038,
            XSDNode::Boolean => 60039,
            XSDNode::Entity => 60040,
            XSDNode::UnsignedByte => 60041,
            XSDNode::Byte => 60042,
            XSDNode::HexBinary => 60043,
        }
    }
}

// #[derive(
//     Copy,
//     Clone,
//     Debug,
//     PartialEq,
//     Eq,
//     Hash,
//     Archive,
//     Deserialize,
//     Serialize,
//     EnumIter,
//     strum::Display,
// )]
// #[strum(serialize_all = "title_case")]
// pub enum XSDEdge {

// }

// impl XSDEdge {
//     #[cfg(feature = "test-utils")]
//     pub(crate) const fn sovs_kind(self) -> &'static str {
//         match self {

//         }
//     }
// }

// impl From<XSDEdge> for u32 {
//     fn from(value: XSDEdge) -> Self {
//         match value {

//         }
//     }
// }
