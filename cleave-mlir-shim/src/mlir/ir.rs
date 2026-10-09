//! IR objects: locations, types, attributes, values, operations, blocks,
//! regions, modules.

use super::{Context, Error, StringRef, printed, sys};
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

pub use attribute::Attribute;
pub use block::{Block, BlockRef};
pub use operation::{Operation, OperationRef};
pub use r#type::Type;
pub use region::{Region, RegionRef};

/// Where an operation comes from.
#[derive(Clone, Copy)]
pub struct Location<'c> {
    raw: sys::MlirLocation,
    _context: PhantomData<&'c Context>,
}

impl<'c> Location<'c> {
    pub fn new(context: &'c Context, filename: &str, line: usize, column: usize) -> Self {
        unsafe {
            Self::from_raw(sys::mlirLocationFileLineColGet(
                context.to_raw(),
                StringRef::new(filename).to_raw(),
                line as u32,
                column as u32,
            ))
        }
    }

    pub fn unknown(context: &'c Context) -> Self {
        unsafe { Self::from_raw(sys::mlirLocationUnknownGet(context.to_raw())) }
    }

    pub fn fused(context: &'c Context, locations: &[Self], metadata: Attribute) -> Self {
        let raw: Vec<sys::MlirLocation> = locations.iter().map(|l| l.raw).collect();
        unsafe {
            Self::from_raw(sys::mlirLocationFusedGet(context.to_raw(), raw.len() as isize, raw.as_ptr(), metadata.to_raw()))
        }
    }

    /// # Safety
    ///
    /// `raw` must be a valid location of a context living as long as `'c`.
    pub unsafe fn from_raw(raw: sys::MlirLocation) -> Self {
        Self { raw, _context: PhantomData }
    }

    pub fn to_raw(&self) -> sys::MlirLocation {
        self.raw
    }
}

impl fmt::Display for Location<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&printed(|cb, ud| unsafe { sys::mlirLocationPrint(self.raw, cb, ud) }))
    }
}

/// An interned name (an attribute's, an operation's).
#[derive(Clone, Copy)]
pub struct Identifier<'c> {
    raw: sys::MlirIdentifier,
    _context: PhantomData<&'c Context>,
}

impl<'c> Identifier<'c> {
    pub fn new(context: &'c Context, name: &str) -> Self {
        Self {
            raw: unsafe { sys::mlirIdentifierGet(context.to_raw(), StringRef::new(name).to_raw()) },
            _context: PhantomData,
        }
    }

    /// # Safety
    ///
    /// `raw` must be a valid identifier of a context living as long as `'c`.
    pub unsafe fn from_raw(raw: sys::MlirIdentifier) -> Self {
        Self { raw, _context: PhantomData }
    }

    pub fn to_raw(&self) -> sys::MlirIdentifier {
        self.raw
    }

    pub fn as_string_ref(&self) -> StringRef<'c> {
        unsafe { StringRef::from_raw(sys::mlirIdentifierStr(self.raw)) }
    }
}

pub mod r#type {
    use super::*;

    /// A type.
    #[derive(Clone, Copy)]
    pub struct Type<'c> {
        raw: sys::MlirType,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> Type<'c> {
        pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
            unsafe { Self::from_option_raw(sys::mlirTypeParseGet(context.to_raw(), StringRef::new(source).to_raw())) }
        }

        pub fn index(context: &'c Context) -> Self {
            unsafe { Self::from_raw(sys::mlirIndexTypeGet(context.to_raw())) }
        }

        /// # Safety
        ///
        /// `raw` must be a valid type of a context living as long as `'c`.
        pub unsafe fn from_raw(raw: sys::MlirType) -> Self {
            Self { raw, _context: PhantomData }
        }

        /// # Safety
        ///
        /// As `from_raw`, or null.
        pub unsafe fn from_option_raw(raw: sys::MlirType) -> Option<Self> {
            (!raw.ptr.is_null()).then(|| unsafe { Self::from_raw(raw) })
        }

        pub fn to_raw(&self) -> sys::MlirType {
            self.raw
        }

        pub fn is_integer(&self) -> bool {
            unsafe { sys::mlirTypeIsAInteger(self.raw) }
        }
        pub fn is_index(&self) -> bool {
            unsafe { sys::mlirTypeIsAIndex(self.raw) }
        }
        pub fn is_float(&self) -> bool {
            unsafe { sys::mlirTypeIsAFloat(self.raw) }
        }
        pub fn is_mem_ref(&self) -> bool {
            unsafe { sys::mlirTypeIsAMemRef(self.raw) }
        }
        pub fn is_tensor(&self) -> bool {
            unsafe { sys::mlirTypeIsATensor(self.raw) }
        }
        pub fn is_ranked_tensor(&self) -> bool {
            unsafe { sys::mlirTypeIsARankedTensor(self.raw) }
        }
        pub fn is_vector(&self) -> bool {
            unsafe { sys::mlirTypeIsAVector(self.raw) }
        }
        pub fn is_function(&self) -> bool {
            unsafe { sys::mlirTypeIsAFunction(self.raw) }
        }
    }

    impl PartialEq for Type<'_> {
        fn eq(&self, other: &Self) -> bool {
            unsafe { sys::mlirTypeEqual(self.raw, other.raw) }
        }
    }
    impl Eq for Type<'_> {}

    impl fmt::Display for Type<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(&printed(|cb, ud| unsafe { sys::mlirTypePrint(self.raw, cb, ud) }))
        }
    }

    impl fmt::Debug for Type<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Type({self})")
        }
    }

    /// A type of one kind, convertible to and from `Type`.
    macro_rules! typed {
        ($name:ident, $is:ident) => {
            #[derive(Clone, Copy, PartialEq, Eq)]
            pub struct $name<'c> {
                r#type: Type<'c>,
            }

            impl<'c> $name<'c> {
                /// # Safety
                ///
                /// `raw` must be a valid type of this kind.
                pub unsafe fn from_raw(raw: sys::MlirType) -> Self {
                    Self { r#type: unsafe { Type::from_raw(raw) } }
                }
                pub fn to_raw(&self) -> sys::MlirType {
                    self.r#type.to_raw()
                }
            }

            impl<'c> From<$name<'c>> for Type<'c> {
                fn from(t: $name<'c>) -> Self {
                    t.r#type
                }
            }

            impl<'c> TryFrom<Type<'c>> for $name<'c> {
                type Error = Error;
                fn try_from(t: Type<'c>) -> Result<Self, Error> {
                    if unsafe { sys::$is(t.to_raw()) } {
                        Ok(Self { r#type: t })
                    } else {
                        Err(Error(format!("`{t}` isn't a {}", stringify!($name))))
                    }
                }
            }

            impl fmt::Display for $name<'_> {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Display::fmt(&self.r#type, f)
                }
            }

            impl<'c> Deref for $name<'c> {
                type Target = Type<'c>;
                fn deref(&self) -> &Type<'c> {
                    &self.r#type
                }
            }
        };
    }

    typed!(IntegerType, mlirTypeIsAInteger);
    typed!(FunctionType, mlirTypeIsAFunction);
    typed!(MemRefType, mlirTypeIsAMemRef);
    typed!(RankedTensorType, mlirTypeIsARankedTensor);

    impl<'c> IntegerType<'c> {
        pub fn new(context: &'c Context, bits: u32) -> Self {
            unsafe { Self::from_raw(sys::mlirIntegerTypeGet(context.to_raw(), bits)) }
        }
        pub fn width(&self) -> u32 {
            unsafe { sys::mlirIntegerTypeGetWidth(self.to_raw()) }
        }
    }

    impl<'c> FunctionType<'c> {
        pub fn new(context: &'c Context, inputs: &[Type<'c>], results: &[Type<'c>]) -> Self {
            let inputs: Vec<sys::MlirType> = inputs.iter().map(Type::to_raw).collect();
            let results: Vec<sys::MlirType> = results.iter().map(Type::to_raw).collect();
            unsafe {
                Self::from_raw(sys::mlirFunctionTypeGet(
                    context.to_raw(),
                    inputs.len() as isize,
                    inputs.as_ptr(),
                    results.len() as isize,
                    results.as_ptr(),
                ))
            }
        }
    }

    impl<'c> MemRefType<'c> {
        /// `layout` and `memory_space` `None` for the default (identity, 0).
        pub fn new(
            element: Type<'c>,
            dimensions: &[i64],
            layout: Option<Attribute<'c>>,
            memory_space: Option<Attribute<'c>>,
        ) -> Self {
            let null = || unsafe { Attribute::from_raw(sys::mlirAttributeGetNull()) };
            unsafe {
                Self::from_raw(sys::mlirMemRefTypeGet(
                    element.to_raw(),
                    dimensions.len() as isize,
                    dimensions.as_ptr(),
                    layout.unwrap_or_else(null).to_raw(),
                    memory_space.unwrap_or_else(null).to_raw(),
                ))
            }
        }
    }

    /// A dimension's size: known, or only at run time.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum DimSize {
        Static(u64),
        Dynamic,
    }

    /// What a shaped type (a memref, a tensor) has: an element type, a rank,
    /// dimension sizes.
    macro_rules! shaped {
        ($name:ident) => {
            impl<'c> $name<'c> {
                pub fn element(&self) -> Type<'c> {
                    unsafe { Type::from_raw(sys::mlirShapedTypeGetElementType(self.to_raw())) }
                }
                pub fn rank(&self) -> usize {
                    unsafe { sys::mlirShapedTypeGetRank(self.to_raw()) as usize }
                }
                pub fn dim_size(&self, index: usize) -> Result<DimSize, Error> {
                    if index >= self.rank() {
                        return Err(Error(format!("dimension {index} out of `{self}`'s rank")));
                    }
                    let size = unsafe { sys::mlirShapedTypeGetDimSize(self.to_raw(), index as isize) };
                    Ok(if unsafe { sys::mlirShapedTypeIsDynamicSize(size) } {
                        DimSize::Dynamic
                    } else {
                        DimSize::Static(size as u64)
                    })
                }
            }
        };
    }
    shaped!(MemRefType);
    shaped!(RankedTensorType);
}

pub mod attribute {
    use super::*;

    /// An attribute.
    #[derive(Clone, Copy)]
    pub struct Attribute<'c> {
        raw: sys::MlirAttribute,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> Attribute<'c> {
        pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
            let raw = unsafe { sys::mlirAttributeParseGet(context.to_raw(), StringRef::new(source).to_raw()) };
            (!raw.ptr.is_null()).then(|| unsafe { Self::from_raw(raw) })
        }

        pub fn unit(context: &'c Context) -> Self {
            unsafe { Self::from_raw(sys::mlirUnitAttrGet(context.to_raw())) }
        }

        /// # Safety
        ///
        /// `raw` must be a valid attribute of a context living as long as
        /// `'c`, or null.
        pub unsafe fn from_raw(raw: sys::MlirAttribute) -> Self {
            Self { raw, _context: PhantomData }
        }

        pub fn to_raw(&self) -> sys::MlirAttribute {
            self.raw
        }

        pub fn is_null(&self) -> bool {
            self.raw.ptr.is_null()
        }
    }

    impl PartialEq for Attribute<'_> {
        fn eq(&self, other: &Self) -> bool {
            unsafe { sys::mlirAttributeEqual(self.raw, other.raw) }
        }
    }
    impl Eq for Attribute<'_> {}

    impl fmt::Display for Attribute<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(&printed(|cb, ud| unsafe { sys::mlirAttributePrint(self.raw, cb, ud) }))
        }
    }

    impl fmt::Debug for Attribute<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Attribute({self})")
        }
    }

    macro_rules! typed {
        ($name:ident) => {
            #[derive(Clone, Copy)]
            pub struct $name<'c> {
                attribute: Attribute<'c>,
            }

            impl<'c> $name<'c> {
                /// # Safety
                ///
                /// `raw` must be a valid attribute of this kind.
                pub unsafe fn from_raw(raw: sys::MlirAttribute) -> Self {
                    Self { attribute: unsafe { Attribute::from_raw(raw) } }
                }
                pub fn to_raw(&self) -> sys::MlirAttribute {
                    self.attribute.to_raw()
                }
            }

            impl<'c> From<$name<'c>> for Attribute<'c> {
                fn from(a: $name<'c>) -> Self {
                    a.attribute
                }
            }

            impl fmt::Display for $name<'_> {
                fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                    fmt::Display::fmt(&self.attribute, f)
                }
            }
        };
    }

    typed!(IntegerAttribute);
    typed!(FloatAttribute);
    typed!(StringAttribute);
    typed!(FlatSymbolRefAttribute);
    typed!(TypeAttribute);
    typed!(DenseI32ArrayAttribute);
    typed!(DenseI64ArrayAttribute);
    typed!(ArrayAttribute);

    impl<'c> IntegerAttribute<'c> {
        pub fn new(r#type: Type<'c>, value: i64) -> Self {
            unsafe { Self::from_raw(sys::mlirIntegerAttrGet(r#type.to_raw(), value)) }
        }
    }

    impl<'c> FloatAttribute<'c> {
        pub fn new(context: &'c Context, r#type: Type<'c>, value: f64) -> Self {
            unsafe { Self::from_raw(sys::mlirFloatAttrDoubleGet(context.to_raw(), r#type.to_raw(), value)) }
        }
    }

    impl<'c> StringAttribute<'c> {
        pub fn new(context: &'c Context, string: &str) -> Self {
            unsafe { Self::from_raw(sys::mlirStringAttrGet(context.to_raw(), StringRef::new(string).to_raw())) }
        }
    }

    impl<'c> FlatSymbolRefAttribute<'c> {
        pub fn new(context: &'c Context, symbol: &str) -> Self {
            unsafe { Self::from_raw(sys::mlirFlatSymbolRefAttrGet(context.to_raw(), StringRef::new(symbol).to_raw())) }
        }
    }

    impl<'c> TypeAttribute<'c> {
        pub fn new(r#type: Type<'c>) -> Self {
            unsafe { Self::from_raw(sys::mlirTypeAttrGet(r#type.to_raw())) }
        }
    }

    impl<'c> DenseI32ArrayAttribute<'c> {
        pub fn new(context: &'c Context, values: &[i32]) -> Self {
            unsafe { Self::from_raw(sys::mlirDenseI32ArrayGet(context.to_raw(), values.len() as isize, values.as_ptr())) }
        }
    }

    impl<'c> DenseI64ArrayAttribute<'c> {
        pub fn new(context: &'c Context, values: &[i64]) -> Self {
            unsafe { Self::from_raw(sys::mlirDenseI64ArrayGet(context.to_raw(), values.len() as isize, values.as_ptr())) }
        }
    }

    impl<'c> ArrayAttribute<'c> {
        /// An array attribute's elements, `attribute` an array attribute.
        ///
        /// # Safety
        ///
        /// `attribute` must be an `ArrayAttr`.
        pub unsafe fn elements_of(attribute: Attribute<'c>) -> Vec<Attribute<'c>> {
            let raw = attribute.to_raw();
            let n = unsafe { sys::mlirArrayAttrGetNumElements(raw) };
            (0..n).map(|i| unsafe { Attribute::from_raw(sys::mlirArrayAttrGetElement(raw, i)) }).collect()
        }
    }
}

pub mod value {
    use super::*;

    /// A value: an operation's result or a block's argument. `'a` is how
    /// long its owner is borrowed for.
    #[derive(Clone, Copy)]
    pub struct Value<'c, 'a> {
        raw: sys::MlirValue,
        _owner: PhantomData<(&'c Context, &'a ())>,
    }

    impl<'c, 'a> Value<'c, 'a> {
        /// # Safety
        ///
        /// `raw` must be a valid value whose owner lives as long as `'a`.
        pub unsafe fn from_raw(raw: sys::MlirValue) -> Self {
            Self { raw, _owner: PhantomData }
        }

        pub fn to_raw(&self) -> sys::MlirValue {
            self.raw
        }

        pub fn r#type(&self) -> Type<'c> {
            unsafe { Type::from_raw(sys::mlirValueGetType(self.raw)) }
        }

        pub fn is_operation_result(&self) -> bool {
            unsafe { sys::mlirValueIsAOpResult(self.raw) }
        }

        pub fn is_block_argument(&self) -> bool {
            unsafe { sys::mlirValueIsABlockArgument(self.raw) }
        }
    }

    impl PartialEq for Value<'_, '_> {
        fn eq(&self, other: &Self) -> bool {
            self.raw.ptr == other.raw.ptr
        }
    }
    impl Eq for Value<'_, '_> {}

    impl fmt::Display for Value<'_, '_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(&printed(|cb, ud| unsafe { sys::mlirValuePrint(self.raw, cb, ud) }))
        }
    }

    impl fmt::Debug for Value<'_, '_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Value({self})")
        }
    }

    /// A value that is an operation's result.
    #[derive(Clone, Copy)]
    pub struct OperationResult<'c, 'a> {
        value: Value<'c, 'a>,
    }

    impl<'c, 'a> OperationResult<'c, 'a> {
        pub fn owner(&self) -> OperationRef<'c, 'a> {
            unsafe { OperationRef::from_raw(sys::mlirOpResultGetOwner(self.value.to_raw())) }
        }

        pub fn result_number(&self) -> usize {
            unsafe { sys::mlirOpResultGetResultNumber(self.value.to_raw()) as usize }
        }
    }

    impl<'c, 'a> From<OperationResult<'c, 'a>> for Value<'c, 'a> {
        fn from(r: OperationResult<'c, 'a>) -> Self {
            r.value
        }
    }

    impl<'c, 'a> TryFrom<Value<'c, 'a>> for OperationResult<'c, 'a> {
        type Error = Error;
        fn try_from(value: Value<'c, 'a>) -> Result<Self, Error> {
            if value.is_operation_result() {
                Ok(Self { value })
            } else {
                Err(Error("not an operation's result".into()))
            }
        }
    }

    impl<'c, 'a> Deref for OperationResult<'c, 'a> {
        type Target = Value<'c, 'a>;
        fn deref(&self) -> &Value<'c, 'a> {
            &self.value
        }
    }

    /// A value that is a block's argument.
    #[derive(Clone, Copy)]
    pub struct BlockArgument<'c, 'a> {
        value: Value<'c, 'a>,
    }

    impl<'c, 'a> From<BlockArgument<'c, 'a>> for Value<'c, 'a> {
        fn from(a: BlockArgument<'c, 'a>) -> Self {
            a.value
        }
    }

    impl<'c, 'a> Deref for BlockArgument<'c, 'a> {
        type Target = Value<'c, 'a>;
        fn deref(&self) -> &Value<'c, 'a> {
            &self.value
        }
    }

    impl<'c, 'a> BlockArgument<'c, 'a> {
        pub(crate) fn new(value: Value<'c, 'a>) -> Self {
            Self { value }
        }
    }

    impl<'c, 'a> OperationResult<'c, 'a> {
        pub(crate) fn new(value: Value<'c, 'a>) -> Self {
            Self { value }
        }
    }
}

pub use value::Value;

pub mod operation {
    use super::*;
    pub use crate::mlir::ir::value::OperationResult;
    use crate::mlir::ir::value::Value;

    /// An operation not inserted anywhere: owned, destroyed on drop.
    pub struct Operation<'c> {
        raw: sys::MlirOperation,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> Operation<'c> {
        /// # Safety
        ///
        /// `raw` must be a valid operation the caller owns.
        pub unsafe fn from_raw(raw: sys::MlirOperation) -> Self {
            Self { raw, _context: PhantomData }
        }

        /// # Safety
        ///
        /// As `from_raw`, or null.
        pub unsafe fn from_option_raw(raw: sys::MlirOperation) -> Option<Self> {
            (!raw.ptr.is_null()).then(|| unsafe { Self::from_raw(raw) })
        }

        /// The handle, ownership kept.
        pub fn to_raw(&self) -> sys::MlirOperation {
            self.raw
        }

        /// The handle, ownership handed over to the caller.
        pub fn into_raw(self) -> sys::MlirOperation {
            let raw = self.raw;
            std::mem::forget(self);
            raw
        }

        pub fn name(&self) -> Identifier<'c> {
            unsafe { Identifier::from_raw(sys::mlirOperationGetName(self.raw)) }
        }

        pub fn location(&self) -> Location<'c> {
            unsafe { Location::from_raw(sys::mlirOperationGetLocation(self.raw)) }
        }

        pub fn set_location(&self, location: Location<'c>) {
            unsafe { sys::mlirOperationSetLocation(self.raw, location.to_raw()) }
        }

        pub fn block<'a>(&self) -> Option<BlockRef<'c, 'a>> where 'c: 'a {
            unsafe { BlockRef::from_option_raw(sys::mlirOperationGetBlock(self.raw)) }
        }

        pub fn parent_operation<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            unsafe { OperationRef::from_option_raw(sys::mlirOperationGetParentOperation(self.raw)) }
        }

        pub fn next_in_block<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            unsafe { OperationRef::from_option_raw(sys::mlirOperationGetNextInBlock(self.raw)) }
        }

        pub fn region_count(&self) -> usize {
            unsafe { sys::mlirOperationGetNumRegions(self.raw) as usize }
        }

        pub fn region<'a>(&self, index: usize) -> Result<RegionRef<'c, 'a>, Error> where 'c: 'a {
            if index < self.region_count() {
                Ok(unsafe { RegionRef::from_raw(sys::mlirOperationGetRegion(self.raw, index as isize)) })
            } else {
                Err(Error(format!("no region {index}")))
            }
        }

        pub fn regions<'a>(&self) -> impl Iterator<Item = RegionRef<'c, 'a>> where 'c: 'a {
            (0..self.region_count())
                .map(move |i| unsafe { RegionRef::from_raw(sys::mlirOperationGetRegion(self.raw, i as isize)) })
        }

        pub fn result_count(&self) -> usize {
            unsafe { sys::mlirOperationGetNumResults(self.raw) as usize }
        }

        pub fn result<'a>(&self, index: usize) -> Result<OperationResult<'c, 'a>, Error> where 'c: 'a {
            if index < self.result_count() {
                Ok(OperationResult::new(unsafe {
                    Value::from_raw(sys::mlirOperationGetResult(self.raw, index as isize))
                }))
            } else {
                Err(Error(format!("no result {index}")))
            }
        }

        pub fn operand_count(&self) -> usize {
            unsafe { sys::mlirOperationGetNumOperands(self.raw) as usize }
        }

        pub fn operand<'a>(&self, index: usize) -> Result<Value<'c, 'a>, Error> where 'c: 'a {
            if index < self.operand_count() {
                Ok(unsafe { Value::from_raw(sys::mlirOperationGetOperand(self.raw, index as isize)) })
            } else {
                Err(Error(format!("no operand {index}")))
            }
        }

        pub fn set_operand<'a>(&self, index: usize, value: Value<'c, 'a>) where 'c: 'a {
            unsafe { sys::mlirOperationSetOperand(self.raw, index as isize, value.to_raw()) }
        }

        pub fn attribute(&self, name: &str) -> Result<Attribute<'c>, Error> {
            let raw = unsafe { sys::mlirOperationGetAttributeByName(self.raw, StringRef::new(name).to_raw()) };
            if raw.ptr.is_null() { Err(Error(format!("no attribute `{name}`"))) } else { Ok(unsafe { Attribute::from_raw(raw) }) }
        }

        pub fn set_attribute(&self, name: &str, attribute: Attribute<'c>) {
            unsafe { sys::mlirOperationSetAttributeByName(self.raw, StringRef::new(name).to_raw(), attribute.to_raw()) }
        }

        pub fn set_discardable_attribute(&self, name: &str, attribute: Attribute<'c>) {
            unsafe {
                sys::mlirOperationSetDiscardableAttributeByName(self.raw, StringRef::new(name).to_raw(), attribute.to_raw())
            }
        }

        pub fn set_operands(&self, values: &[Value<'c, '_>]) {
            let raw: Vec<sys::MlirValue> = values.iter().map(Value::to_raw).collect();
            unsafe { sys::mlirOperationSetOperands(self.raw, raw.len() as isize, raw.as_ptr()) }
        }

        pub fn remove_attribute(&self, name: &str) -> bool {
            unsafe { sys::mlirOperationRemoveAttributeByName(self.raw, StringRef::new(name).to_raw()) }
        }

        pub fn verify(&self) -> bool {
            unsafe { sys::mlirOperationVerify(self.raw) }
        }

        /// Unlinks the operation from its block. The caller owns it then:
        /// `Operation::from_raw` it, to re-insert or drop it.
        pub fn remove_from_parent(&self) {
            unsafe { sys::mlirOperationRemoveFromParent(self.raw) }
        }

        pub fn move_before<'a>(&self, other: OperationRef<'c, 'a>) where 'c: 'a {
            unsafe { sys::mlirOperationMoveBefore(self.raw, other.to_raw()) }
        }

        pub fn move_after<'a>(&self, other: OperationRef<'c, 'a>) where 'c: 'a {
            unsafe { sys::mlirOperationMoveAfter(self.raw, other.to_raw()) }
        }
    }

    impl Drop for Operation<'_> {
        fn drop(&mut self) {
            unsafe { sys::mlirOperationDestroy(self.raw) }
        }
    }

    impl fmt::Display for Operation<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str(&printed(|cb, ud| unsafe { sys::mlirOperationPrint(self.raw, cb, ud) }))
        }
    }

    impl fmt::Debug for Operation<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "Operation({self})")
        }
    }

    /// An operation owned by its block, borrowed for `'a`.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct OperationRef<'c, 'a> {
        raw: sys::MlirOperation,
        _owner: PhantomData<(&'c Context, &'a ())>,
    }

    impl<'c, 'a> OperationRef<'c, 'a> {
        /// # Safety
        ///
        /// `raw` must be a valid operation owned by something living `'a`.
        pub unsafe fn from_raw(raw: sys::MlirOperation) -> Self {
            Self { raw, _owner: PhantomData }
        }

        /// # Safety
        ///
        /// As `from_raw`, or null.
        pub unsafe fn from_option_raw(raw: sys::MlirOperation) -> Option<Self> {
            (!raw.ptr.is_null()).then(|| unsafe { Self::from_raw(raw) })
        }

        pub fn to_raw(&self) -> sys::MlirOperation {
            self.raw
        }

        pub fn result(&self, index: usize) -> Result<OperationResult<'c, 'a>, Error> {
            // SAFETY: the result lives as long as the operation, `'a`.
            let r = self.deref().result(index)?;
            Ok(OperationResult::new(unsafe { Value::from_raw(r.to_raw()) }))
        }

        pub fn operand(&self, index: usize) -> Result<Value<'c, 'a>, Error> {
            let v = self.deref().operand(index)?;
            Ok(unsafe { Value::from_raw(v.to_raw()) })
        }

        pub fn block(&self) -> Option<BlockRef<'c, 'a>> {
            unsafe { BlockRef::from_option_raw(sys::mlirOperationGetBlock(self.raw)) }
        }

        pub fn next_in_block(&self) -> Option<OperationRef<'c, 'a>> {
            unsafe { OperationRef::from_option_raw(sys::mlirOperationGetNextInBlock(self.raw)) }
        }

        pub fn region(&self, index: usize) -> Result<RegionRef<'c, 'a>, Error> {
            let r = self.deref().region(index)?;
            Ok(unsafe { RegionRef::from_raw(r.to_raw()) })
        }
    }

    impl<'c> Deref for OperationRef<'c, '_> {
        type Target = Operation<'c>;
        fn deref(&self) -> &Operation<'c> {
            // SAFETY: same layout (one raw handle); a reference never drops.
            unsafe { &*(self as *const Self as *const Operation<'c>) }
        }
    }

    impl PartialEq for OperationRef<'_, '_> {
        fn eq(&self, other: &Self) -> bool {
            self.raw.ptr == other.raw.ptr
        }
    }
    impl Eq for OperationRef<'_, '_> {}

    impl fmt::Display for OperationRef<'_, '_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(self.deref(), f)
        }
    }

    /// `OperationRef`, by the name code mutating an operation uses.
    pub type OperationRefMut<'c, 'a> = OperationRef<'c, 'a>;

    /// Builds an operation (`mlirOperationCreate`).
    pub struct OperationBuilder<'c> {
        state: sys::MlirOperationState,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> OperationBuilder<'c> {
        pub fn new(name: &str, location: Location<'c>) -> Self {
            Self {
                state: unsafe { sys::mlirOperationStateGet(StringRef::new(name).to_raw(), location.to_raw()) },
                _context: PhantomData,
            }
        }

        pub fn add_results(mut self, results: &[Type<'c>]) -> Self {
            let raw: Vec<sys::MlirType> = results.iter().map(Type::to_raw).collect();
            unsafe { sys::mlirOperationStateAddResults(&mut self.state, raw.len() as isize, raw.as_ptr()) }
            self
        }

        pub fn add_operands(mut self, operands: &[Value<'c, '_>]) -> Self {
            let raw: Vec<sys::MlirValue> = operands.iter().map(Value::to_raw).collect();
            unsafe { sys::mlirOperationStateAddOperands(&mut self.state, raw.len() as isize, raw.as_ptr()) }
            self
        }

        pub fn add_regions<const N: usize>(self, regions: [Region<'c>; N]) -> Self {
            self.add_regions_vec(regions.into())
        }

        pub fn add_regions_vec(mut self, regions: Vec<Region<'c>>) -> Self {
            let raw: Vec<sys::MlirRegion> = regions.into_iter().map(Region::into_raw).collect();
            unsafe { sys::mlirOperationStateAddOwnedRegions(&mut self.state, raw.len() as isize, raw.as_ptr()) }
            self
        }

        pub fn add_successors(mut self, successors: &[&Block<'c>]) -> Self {
            let raw: Vec<sys::MlirBlock> = successors.iter().map(|b| b.to_raw()).collect();
            unsafe { sys::mlirOperationStateAddSuccessors(&mut self.state, raw.len() as isize, raw.as_ptr()) }
            self
        }

        pub fn add_attributes(mut self, attributes: &[(Identifier<'c>, Attribute<'c>)]) -> Self {
            let raw: Vec<sys::MlirNamedAttribute> = attributes
                .iter()
                .map(|(name, attribute)| sys::MlirNamedAttribute { name: name.to_raw(), attribute: attribute.to_raw() })
                .collect();
            unsafe { sys::mlirOperationStateAddAttributes(&mut self.state, raw.len() as isize, raw.as_ptr()) }
            self
        }

        pub fn enable_result_type_inference(mut self) -> Self {
            unsafe { sys::mlirOperationStateEnableResultTypeInference(&mut self.state) }
            self
        }

        pub fn build(mut self) -> Result<Operation<'c>, Error> {
            let name = unsafe { StringRef::from_raw(self.state.name) }.as_str().unwrap_or("?").to_string();
            unsafe { Operation::from_option_raw(sys::mlirOperationCreate(&mut self.state)) }
                .ok_or_else(|| Error(format!("failed to build `{name}`")))
        }
    }
}

pub mod block {
    use super::*;
    use crate::mlir::ir::value::{BlockArgument, Value};

    /// A block not inserted in a region: owned, destroyed on drop.
    pub struct Block<'c> {
        raw: sys::MlirBlock,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> Block<'c> {
        pub fn new(arguments: &[(Type<'c>, Location<'c>)]) -> Self {
            let types: Vec<sys::MlirType> = arguments.iter().map(|(t, _)| t.to_raw()).collect();
            let locations: Vec<sys::MlirLocation> = arguments.iter().map(|(_, l)| l.to_raw()).collect();
            Self {
                raw: unsafe { sys::mlirBlockCreate(types.len() as isize, types.as_ptr(), locations.as_ptr()) },
                _context: PhantomData,
            }
        }

        pub fn to_raw(&self) -> sys::MlirBlock {
            self.raw
        }

        pub fn into_raw(self) -> sys::MlirBlock {
            let raw = self.raw;
            std::mem::forget(self);
            raw
        }

        pub fn argument_count(&self) -> usize {
            unsafe { sys::mlirBlockGetNumArguments(self.raw) as usize }
        }

        pub fn argument<'a>(&self, index: usize) -> Result<BlockArgument<'c, 'a>, Error> where 'c: 'a {
            if index < self.argument_count() {
                Ok(BlockArgument::new(unsafe { Value::from_raw(sys::mlirBlockGetArgument(self.raw, index as isize)) }))
            } else {
                Err(Error(format!("no argument {index}")))
            }
        }

        pub fn add_argument<'a>(&self, r#type: Type<'c>, location: Location<'c>) -> Value<'c, 'a> where 'c: 'a {
            unsafe { Value::from_raw(sys::mlirBlockAddArgument(self.raw, r#type.to_raw(), location.to_raw())) }
        }

        pub fn parent_operation<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            unsafe { OperationRef::from_option_raw(sys::mlirBlockGetParentOperation(self.raw)) }
        }

        pub fn next_in_region<'a>(&self) -> Option<BlockRef<'c, 'a>> where 'c: 'a {
            unsafe { BlockRef::from_option_raw(sys::mlirBlockGetNextInRegion(self.raw)) }
        }

        pub fn first_operation<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            unsafe { OperationRef::from_option_raw(sys::mlirBlockGetFirstOperation(self.raw)) }
        }

        pub fn first_operation_mut<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            self.first_operation()
        }

        pub fn terminator<'a>(&self) -> Option<OperationRef<'c, 'a>> where 'c: 'a {
            unsafe { OperationRef::from_option_raw(sys::mlirBlockGetTerminator(self.raw)) }
        }

        pub fn append_operation<'a>(&self, operation: Operation<'c>) -> OperationRef<'c, 'a> where 'c: 'a {
            let raw = operation.into_raw();
            unsafe {
                sys::mlirBlockAppendOwnedOperation(self.raw, raw);
                OperationRef::from_raw(raw)
            }
        }

        pub fn insert_operation<'a>(&self, position: usize, operation: Operation<'c>) -> OperationRef<'c, 'a> where 'c: 'a {
            let raw = operation.into_raw();
            unsafe {
                sys::mlirBlockInsertOwnedOperation(self.raw, position as isize, raw);
                OperationRef::from_raw(raw)
            }
        }

        pub fn insert_operation_after<'a>(
            &self,
            reference: OperationRef<'c, 'a>,
            operation: Operation<'c>,
        ) -> OperationRef<'c, 'a> where 'c: 'a {
            let raw = operation.into_raw();
            unsafe {
                sys::mlirBlockInsertOwnedOperationAfter(self.raw, reference.to_raw(), raw);
                OperationRef::from_raw(raw)
            }
        }

        pub fn insert_operation_before<'a>(
            &self,
            reference: OperationRef<'c, 'a>,
            operation: Operation<'c>,
        ) -> OperationRef<'c, 'a> where 'c: 'a {
            let raw = operation.into_raw();
            unsafe {
                sys::mlirBlockInsertOwnedOperationBefore(self.raw, reference.to_raw(), raw);
                OperationRef::from_raw(raw)
            }
        }
    }

    impl Drop for Block<'_> {
        fn drop(&mut self) {
            unsafe { sys::mlirBlockDestroy(self.raw) }
        }
    }

    /// A block owned by its region, borrowed for `'a`.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct BlockRef<'c, 'a> {
        raw: sys::MlirBlock,
        _owner: PhantomData<(&'c Context, &'a ())>,
    }

    impl<'c, 'a> BlockRef<'c, 'a> {
        /// # Safety
        ///
        /// `raw` must be a valid block owned by something living `'a`.
        pub unsafe fn from_raw(raw: sys::MlirBlock) -> Self {
            Self { raw, _owner: PhantomData }
        }

        /// # Safety
        ///
        /// As `from_raw`, or null.
        pub unsafe fn from_option_raw(raw: sys::MlirBlock) -> Option<Self> {
            (!raw.ptr.is_null()).then(|| unsafe { Self::from_raw(raw) })
        }

        pub fn to_raw(&self) -> sys::MlirBlock {
            self.raw
        }

        pub fn first_operation(&self) -> Option<OperationRef<'c, 'a>> {
            unsafe { OperationRef::from_option_raw(sys::mlirBlockGetFirstOperation(self.raw)) }
        }

        pub fn next_in_region(&self) -> Option<BlockRef<'c, 'a>> {
            unsafe { BlockRef::from_option_raw(sys::mlirBlockGetNextInRegion(self.raw)) }
        }

        pub fn argument(&self, index: usize) -> Result<BlockArgument<'c, 'a>, Error> {
            let a = self.deref().argument(index)?;
            Ok(BlockArgument::new(unsafe { Value::from_raw(a.to_raw()) }))
        }

        pub fn append_operation(&self, operation: Operation<'c>) -> OperationRef<'c, 'a> {
            let r = self.deref().append_operation(operation);
            unsafe { OperationRef::from_raw(r.to_raw()) }
        }
    }

    impl<'c> Deref for BlockRef<'c, '_> {
        type Target = Block<'c>;
        fn deref(&self) -> &Block<'c> {
            // SAFETY: same layout (one raw handle); a reference never drops.
            unsafe { &*(self as *const Self as *const Block<'c>) }
        }
    }

    impl PartialEq for BlockRef<'_, '_> {
        fn eq(&self, other: &Self) -> bool {
            self.raw.ptr == other.raw.ptr
        }
    }
    impl Eq for BlockRef<'_, '_> {}
}

pub mod region {
    use super::*;

    /// A region not attached to an operation: owned, destroyed on drop.
    pub struct Region<'c> {
        raw: sys::MlirRegion,
        _context: PhantomData<&'c Context>,
    }

    impl<'c> Region<'c> {
        pub fn new() -> Self {
            Self { raw: unsafe { sys::mlirRegionCreate() }, _context: PhantomData }
        }

        pub fn to_raw(&self) -> sys::MlirRegion {
            self.raw
        }

        pub fn into_raw(self) -> sys::MlirRegion {
            let raw = self.raw;
            std::mem::forget(self);
            raw
        }

        pub fn append_block<'a>(&self, block: Block<'c>) -> BlockRef<'c, 'a> where 'c: 'a {
            let raw = block.into_raw();
            unsafe {
                sys::mlirRegionAppendOwnedBlock(self.raw, raw);
                BlockRef::from_raw(raw)
            }
        }

        pub fn first_block<'a>(&self) -> Option<BlockRef<'c, 'a>> where 'c: 'a {
            unsafe { BlockRef::from_option_raw(sys::mlirRegionGetFirstBlock(self.raw)) }
        }
    }

    impl Default for Region<'_> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for Region<'_> {
        fn drop(&mut self) {
            unsafe { sys::mlirRegionDestroy(self.raw) }
        }
    }

    /// A region owned by its operation, borrowed for `'a`.
    #[derive(Clone, Copy)]
    #[repr(transparent)]
    pub struct RegionRef<'c, 'a> {
        raw: sys::MlirRegion,
        _owner: PhantomData<(&'c Context, &'a ())>,
    }

    impl<'c, 'a> RegionRef<'c, 'a> {
        /// # Safety
        ///
        /// `raw` must be a valid region owned by something living `'a`.
        pub unsafe fn from_raw(raw: sys::MlirRegion) -> Self {
            Self { raw, _owner: PhantomData }
        }

        pub fn to_raw(&self) -> sys::MlirRegion {
            self.raw
        }

        pub fn first_block(&self) -> Option<BlockRef<'c, 'a>> {
            unsafe { BlockRef::from_option_raw(sys::mlirRegionGetFirstBlock(self.raw)) }
        }
    }

    impl<'c> Deref for RegionRef<'c, '_> {
        type Target = Region<'c>;
        fn deref(&self) -> &Region<'c> {
            // SAFETY: same layout (one raw handle); a reference never drops.
            unsafe { &*(self as *const Self as *const Region<'c>) }
        }
    }
}

/// A module: owns its operation, destroyed on drop.
pub struct Module<'c> {
    raw: sys::MlirModule,
    _context: PhantomData<&'c Context>,
}

impl<'c> Module<'c> {
    pub fn new(location: Location<'c>) -> Self {
        Self { raw: unsafe { sys::mlirModuleCreateEmpty(location.to_raw()) }, _context: PhantomData }
    }

    pub fn parse(context: &'c Context, source: &str) -> Option<Self> {
        let raw = unsafe { sys::mlirModuleCreateParse(context.to_raw(), StringRef::new(source).to_raw()) };
        (!raw.ptr.is_null()).then_some(Self { raw, _context: PhantomData })
    }

    /// # Safety
    ///
    /// `raw` must be a valid module the caller owns.
    pub unsafe fn from_raw(raw: sys::MlirModule) -> Self {
        Self { raw, _context: PhantomData }
    }

    pub fn to_raw(&self) -> sys::MlirModule {
        self.raw
    }

    pub fn body(&self) -> BlockRef<'c, '_> {
        unsafe { BlockRef::from_raw(sys::mlirModuleGetBody(self.raw)) }
    }

    pub fn as_operation(&self) -> OperationRef<'c, '_> {
        unsafe { OperationRef::from_raw(sys::mlirModuleGetOperation(self.raw)) }
    }

    pub fn as_operation_mut(&mut self) -> OperationRef<'c, '_> {
        self.as_operation()
    }
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { sys::mlirModuleDestroy(self.raw) }
    }
}
