# Characteristics

> [!NOTE]
> Non-exhaustive

- `Transitive`
- `FunctionalProperty`
- `InverseFunctionalProperty`
- `ReflexiveProperty`
- `IrreflexiveProperty`
- `SymmetricProperty`
- `AsymmetricProperty`
- `HasKey`

# Elements (nodes and edges)

This document details the conversion of element enums to number codes for all visualized types.

## Reserved [`0` .. `9999`]

Reserved for internal use.

| Code | Type     | Comment                        |
| :--: | :------- | :----------------------------- |
|  0   | `NoDraw` | Element is not shown on screen |

## RDF [`10_000` .. `19_999`]

All RDF elements.

### Nodes [`10_000` .. `14_999`]

Element codes for RDF nodes.

| Code  | Type               | Comment  |
| :---: | :----------------- | :------- |
| 10000 | `rdf:HTML`         | Datatype |
| 10001 | `rdf:PlainLiteral` | Datatype |
| 10002 | `rdf:XMLLiteral`   | Datatype |

### Edges [`15_000` .. `19_999`]

Element codes for RDF edges.

| Code  | Type           | Comment |
| :---: | :------------- | :------ |
| 15000 | `rdf:Property` |         |

## RDFS [`20_000` .. `29_999`]

All RDFS elements.

### Nodes [`20_000` .. `24_999`]

Element codes for RDFS nodes.

| Code  | Type            | Comment |
| :---: | :-------------- | :------ |
| 20000 | `rdfs:Class`    |         |
| 20001 | `rdfs:Literal`  |         |
| 20002 | `rdfs:Resource` |         |
| 20003 | `rdfs:Datatype` |         |

### Edges [`25_000` .. `29_999`]

Element codes for RDFS edges.

| Code  | Type              | Comment |
| :---: | :---------------- | :------ |
| 25000 | `rdfs:subClassOf` |         |

## OWL [`30_000` .. `39_999`]

All OWL elements.

### Nodes [`30_000` .. `34_999`]

Element codes for OWL nodes.

| Code  | Type                  | Comment                                                              |
| :---: | :-------------------- | :------------------------------------------------------------------- |
| 30000 | `AnonymousClass`      | Blank nodes                                                          |
| 30001 | `owl:Class`           |                                                                      |
| 30002 | `owl:complementOf`    |                                                                      |
| 30003 | `owl:DeprecatedClass` |                                                                      |
| 30004 | `ExternalClass`       | Elements whose base URI differs from that of the visualized ontology |
| 30005 | `owl:equivalentClass` |                                                                      |
| 30006 | `owl:disjointUnionOf` |                                                                      |
| 30007 | `owl:intersectionOf`  |                                                                      |
| 30008 | `owl:Thing`           |                                                                      |
| 30009 | `owl:unionOf`         |                                                                      |
| 30010 | `owl:real`            | Datatype                                                             |
| 30011 | `owl:rational`        | Datatype                                                             |

### Edges [`35_000` .. `39_999`]

Element codes for OWL edges.

| Code  | Type                                       | Comment                                                              |
| :---: | :----------------------------------------- | :------------------------------------------------------------------- |
| 35000 | `owl:DatatypeProperty`                     |                                                                      |
| 35001 | `owl:disjointWith`                         |                                                                      |
| 35002 | `owl:DeprecatedProperty`                   |                                                                      |
| 35003 | `ExternalProperty`                         | Elements whose base URI differs from that of the visualized ontology |
| 35004 | `owl:inverseOf`                            |                                                                      |
| 35005 | `owl:ObjectProperty`                       |                                                                      |
| 35006 | `owl:someValuesFrom` / `owl:allValuesFrom` |                                                                      |

## Generic [`40_000` .. `59_999`]

All Generic elements.

### Nodes [`40_000` .. `49_999`]

Element codes for Generic nodes.

| Code  | Type    | Comment |
| :---: | :------ | :------ |
| 40000 | Generic |         |

### Edges [`50_000` .. `59_999`]

Element codes for Generic edges.

| Code  | Type    | Comment |
| :---: | :------ | :------ |
| 50000 | Generic |         |

## XSD [`60_000` .. `69_999`]

All XSD elements.

### Nodes [`60_000` .. `64_999`]

Element codes for XSD nodes.

| Code  | Type                     | Comment |
| :---: | :----------------------- | :------ |
| 60001 | `xsd:int`                |         |
| 60002 | `xsd:integer`            |         |
| 60003 | `xsd:negativeInteger`    |         |
| 60004 | `xsd:nonNegativeInteger` |         |
| 60005 | `xsd:nonPositiveInteger` |         |
| 60006 | `xsd:positiveInteger`    |         |
| 60007 | `xsd:unsignedInt`        |         |
| 60008 | `xsd:unsignedLong`       |         |
| 60009 | `xsd:unsignedShort`      |         |
| 60010 | `xsd:decimal`            |         |
| 60011 | `xsd:float`              |         |
| 60012 | `xsd:double`             |         |
| 60013 | `xsd:short`              |         |
| 60014 | `xsd:long`               |         |
| 60015 | `xsd:date`               |         |
| 60016 | `xsd:dateTime`           |         |
| 60017 | `xsd:dateTimeStamp`      |         |
| 60018 | `xsd:duration`           |         |
| 60019 | `xsd:gDay`               |         |
| 60020 | `xsd:gMonth`             |         |
| 60021 | `xsd:gMonthDay`          |         |
| 60022 | `xsd:gYear`              |         |
| 60023 | `xsd:gYearMonth`         |         |
| 60024 | `xsd:time`               |         |
| 60025 | `xsd:anyURI`             |         |
| 60026 | `xsd:ID`                 |         |
| 60027 | `xsd:IDREF`              |         |
| 60028 | `xsd:language`           |         |
| 60029 | `xsd:NMTOKEN`            |         |
| 60030 | `xsd:Name`               |         |
| 60031 | `xsd:NCName`             |         |
| 60032 | `xsd:QName`              |         |
| 60033 | `xsd:string`             |         |
| 60034 | `xsd:token`              |         |
| 60035 | `xsd:normalizedString`   |         |
| 60036 | `xsd:NOTATION`           |         |
| 60037 | `xsd:anySimpleType`      |         |
| 60038 | `xsd:base64Binary`       |         |
| 60039 | `xsd:boolean`            |         |
| 60040 | `xsd:ENTITY`             |         |
| 60041 | `xsd:unsignedByte`       |         |
| 60042 | `xsd:byte`               |         |
| 60043 | `xsd:hexBinary`          |         |

### Edges [`65_000` .. `69_999`]

Element codes for XSD edges.

| Code | Type | Comment |
| :--: | :--- | :------ |
|      |      |         |
