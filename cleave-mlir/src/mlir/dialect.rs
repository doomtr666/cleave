//! The dialect registry, and builders for the operations cleave's lowering
//! creates often, each the generic `OperationBuilder` with that operation's
//! operands, attributes and results.

use super::sys;

/// The dialects a context can load.
pub struct DialectRegistry {
    raw: sys::MlirDialectRegistry,
}

impl DialectRegistry {
    pub fn new() -> Self {
        Self { raw: unsafe { sys::mlirDialectRegistryCreate() } }
    }

    pub fn to_raw(&self) -> sys::MlirDialectRegistry {
        self.raw
    }
}

impl Default for DialectRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DialectRegistry {
    fn drop(&mut self) {
        unsafe { sys::mlirDialectRegistryDestroy(self.raw) }
    }
}

pub mod arith {
    use crate::mlir::Context;
    use crate::mlir::ir::{Attribute, Identifier, Location, Operation, Type, Value, operation::OperationBuilder};

    pub fn constant<'c>(context: &'c Context, value: Attribute<'c>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(Identifier::new(context, "value"), value)])
            .enable_result_type_inference()
            .build()
            .expect("valid operation")
    }

    fn binary<'c>(name: &str, lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new(name, location)
            .add_operands(&[lhs, rhs])
            .enable_result_type_inference()
            .build()
            .expect("valid operation")
    }

    pub fn addi<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
        binary("arith.addi", lhs, rhs, location)
    }

    pub fn muli<'c>(lhs: Value<'c, '_>, rhs: Value<'c, '_>, location: Location<'c>) -> Operation<'c> {
        binary("arith.muli", lhs, rhs, location)
    }

    pub fn index_cast<'c>(value: Value<'c, '_>, r#type: Type<'c>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("arith.index_cast", location)
            .add_operands(&[value])
            .add_results(&[r#type])
            .build()
            .expect("valid operation")
    }
}

pub mod func {
    use crate::mlir::Context;
    use crate::mlir::ir::attribute::{FlatSymbolRefAttribute, StringAttribute, TypeAttribute};
    use crate::mlir::ir::{Attribute, Identifier, Location, Operation, Region, Type, Value, operation::OperationBuilder};

    pub fn call<'c>(
        context: &'c Context,
        function: FlatSymbolRefAttribute<'c>,
        arguments: &[Value<'c, '_>],
        result_types: &[Type<'c>],
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("func.call", location)
            .add_attributes(&[(Identifier::new(context, "callee"), function.into())])
            .add_operands(arguments)
            .add_results(result_types)
            .build()
            .expect("valid operation")
    }

    pub fn func<'c>(
        context: &'c Context,
        name: StringAttribute<'c>,
        r#type: TypeAttribute<'c>,
        region: Region<'c>,
        attributes: &[(Identifier<'c>, Attribute<'c>)],
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (Identifier::new(context, "sym_name"), name.into()),
                (Identifier::new(context, "function_type"), r#type.into()),
            ])
            .add_attributes(attributes)
            .add_regions([region])
            .build()
            .expect("valid operation")
    }

    pub fn r#return<'c>(operands: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("func.return", location).add_operands(operands).build().expect("valid operation")
    }
}

pub mod scf {
    use crate::mlir::ir::{Location, Operation, Region, Type, Value, operation::OperationBuilder};

    pub fn condition<'c>(condition: Value<'c, '_>, values: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("scf.condition", location)
            .add_operands(&[condition])
            .add_operands(values)
            .build()
            .expect("valid operation")
    }

    pub fn r#if<'c>(
        condition: Value<'c, '_>,
        result_types: &[Type<'c>],
        then_region: Region<'c>,
        else_region: Region<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("scf.if", location)
            .add_operands(&[condition])
            .add_results(result_types)
            .add_regions([then_region, else_region])
            .build()
            .expect("valid operation")
    }

    pub fn r#while<'c>(
        initial_values: &[Value<'c, '_>],
        result_types: &[Type<'c>],
        before_region: Region<'c>,
        after_region: Region<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("scf.while", location)
            .add_operands(initial_values)
            .add_results(result_types)
            .add_regions([before_region, after_region])
            .build()
            .expect("valid operation")
    }

    pub fn r#yield<'c>(values: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("scf.yield", location).add_operands(values).build().expect("valid operation")
    }

    /// `scf.for` without loop-carried values: `region`'s block takes the
    /// induction variable and ends with `scf.yield`.
    pub fn r#for<'c>(
        lower_bound: Value<'c, '_>,
        upper_bound: Value<'c, '_>,
        step: Value<'c, '_>,
        region: Region<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("scf.for", location)
            .add_operands(&[lower_bound, upper_bound, step])
            .add_regions([region])
            .build()
            .expect("valid operation")
    }
}

pub mod memref {
    use crate::mlir::Context;
    use crate::mlir::ir::attribute::{DenseI32ArrayAttribute, DenseI64ArrayAttribute, IntegerAttribute};
    use crate::mlir::ir::r#type::MemRefType;
    use crate::mlir::ir::{Identifier, Location, Operation, Value, operation::OperationBuilder};

    fn allocate<'c>(
        context: &'c Context,
        name: &str,
        r#type: MemRefType<'c>,
        dynamic_sizes: &[Value<'c, '_>],
        symbols: &[Value<'c, '_>],
        alignment: Option<IntegerAttribute<'c>>,
        location: Location<'c>,
    ) -> Operation<'c> {
        let mut builder = OperationBuilder::new(name, location).add_attributes(&[(
            Identifier::new(context, "operand_segment_sizes"),
            DenseI32ArrayAttribute::new(context, &[dynamic_sizes.len() as i32, symbols.len() as i32]).into(),
        )]);
        builder = builder.add_operands(dynamic_sizes).add_operands(symbols);
        if let Some(alignment) = alignment {
            builder = builder.add_attributes(&[(Identifier::new(context, "alignment"), alignment.into())]);
        }
        builder.add_results(&[r#type.into()]).build().expect("valid operation")
    }

    pub fn alloc<'c>(
        context: &'c Context,
        r#type: MemRefType<'c>,
        dynamic_sizes: &[Value<'c, '_>],
        symbols: &[Value<'c, '_>],
        alignment: Option<IntegerAttribute<'c>>,
        location: Location<'c>,
    ) -> Operation<'c> {
        allocate(context, "memref.alloc", r#type, dynamic_sizes, symbols, alignment, location)
    }

    pub fn alloca<'c>(
        context: &'c Context,
        r#type: MemRefType<'c>,
        dynamic_sizes: &[Value<'c, '_>],
        symbols: &[Value<'c, '_>],
        alignment: Option<IntegerAttribute<'c>>,
        location: Location<'c>,
    ) -> Operation<'c> {
        allocate(context, "memref.alloca", r#type, dynamic_sizes, symbols, alignment, location)
    }

    pub fn load<'c>(memref: Value<'c, '_>, indices: &[Value<'c, '_>], location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("memref.load", location)
            .add_operands(&[memref])
            .add_operands(indices)
            .enable_result_type_inference()
            .build()
            .expect("valid operation")
    }

    pub fn store<'c>(
        value: Value<'c, '_>,
        memref: Value<'c, '_>,
        indices: &[Value<'c, '_>],
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("memref.store", location)
            .add_operands(&[value, memref])
            .add_operands(indices)
            .build()
            .expect("valid operation")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn subview<'c>(
        context: &'c Context,
        source: Value<'c, '_>,
        offsets: &[Value<'c, '_>],
        sizes: &[Value<'c, '_>],
        strides: &[Value<'c, '_>],
        static_offsets: &[i64],
        static_sizes: &[i64],
        static_strides: &[i64],
        result_type: MemRefType<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("memref.subview", location)
            .add_operands(&[source])
            .add_operands(offsets)
            .add_operands(sizes)
            .add_operands(strides)
            .add_results(&[result_type.into()])
            .add_attributes(&[
                (
                    Identifier::new(context, "operand_segment_sizes"),
                    DenseI32ArrayAttribute::new(
                        context,
                        &[1, offsets.len() as i32, sizes.len() as i32, strides.len() as i32],
                    )
                    .into(),
                ),
                (Identifier::new(context, "static_offsets"), DenseI64ArrayAttribute::new(context, static_offsets).into()),
                (Identifier::new(context, "static_sizes"), DenseI64ArrayAttribute::new(context, static_sizes).into()),
                (Identifier::new(context, "static_strides"), DenseI64ArrayAttribute::new(context, static_strides).into()),
            ])
            .build()
            .expect("valid operation")
    }
}

pub mod llvm {
    use crate::mlir::Context;
    use crate::mlir::ir::attribute::{DenseI64ArrayAttribute, IntegerAttribute, TypeAttribute};
    use crate::mlir::ir::{Attribute, Identifier, Location, Operation, Type, Value, operation::OperationBuilder};

    pub mod r#type {
        use crate::mlir::Context;
        use crate::mlir::ir::Type;
        use crate::mlir::sys;

        pub fn pointer(context: &Context, address_space: u32) -> Type<'_> {
            unsafe { Type::from_raw(sys::mlirLLVMPointerTypeGet(context.to_raw(), address_space)) }
        }

        pub fn array(r#type: Type, len: u32) -> Type {
            unsafe { Type::from_raw(sys::mlirLLVMArrayTypeGet(r#type.to_raw(), len)) }
        }

        pub fn r#struct<'c>(context: &'c Context, fields: &[Type<'c>], packed: bool) -> Type<'c> {
            let raw: Vec<sys::MlirType> = fields.iter().map(Type::to_raw).collect();
            unsafe {
                Type::from_raw(sys::mlirLLVMStructTypeLiteralGet(context.to_raw(), raw.len() as isize, raw.as_ptr(), packed))
            }
        }
    }

    /// `llvm.load`/`llvm.store`'s optional attributes: none set.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct LoadStoreOptions;

    impl LoadStoreOptions {
        pub fn new() -> Self {
            Self
        }
    }

    /// `llvm.alloca`'s optional attributes.
    #[derive(Default, Clone, Copy)]
    pub struct AllocaOptions<'c> {
        align: Option<IntegerAttribute<'c>>,
        elem_type: Option<TypeAttribute<'c>>,
    }

    impl<'c> AllocaOptions<'c> {
        pub fn new() -> Self {
            Self { align: None, elem_type: None }
        }

        pub fn align(mut self, align: Option<IntegerAttribute<'c>>) -> Self {
            self.align = align;
            self
        }

        pub fn elem_type(mut self, elem_type: Option<TypeAttribute<'c>>) -> Self {
            self.elem_type = elem_type;
            self
        }

        fn into_attributes(self, context: &'c Context) -> Vec<(Identifier<'c>, Attribute<'c>)> {
            let mut attributes = Vec::new();
            if let Some(align) = self.align {
                attributes.push((Identifier::new(context, "alignment"), align.into()));
            }
            if let Some(elem_type) = self.elem_type {
                attributes.push((Identifier::new(context, "elem_type"), elem_type.into()));
            }
            attributes
        }
    }

    pub fn extract_value<'c>(
        context: &'c Context,
        container: Value<'c, '_>,
        position: DenseI64ArrayAttribute<'c>,
        result_type: Type<'c>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.extractvalue", location)
            .add_attributes(&[(Identifier::new(context, "position"), position.into())])
            .add_operands(&[container])
            .add_results(&[result_type])
            .build()
            .expect("valid operation")
    }

    pub fn insert_value<'c>(
        context: &'c Context,
        container: Value<'c, '_>,
        position: DenseI64ArrayAttribute<'c>,
        value: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.insertvalue", location)
            .add_attributes(&[(Identifier::new(context, "position"), position.into())])
            .add_operands(&[container, value])
            .enable_result_type_inference()
            .build()
            .expect("valid operation")
    }

    pub fn undef<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("llvm.mlir.undef", location).add_results(&[result_type]).build().expect("valid operation")
    }

    pub fn poison<'c>(result_type: Type<'c>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("llvm.mlir.poison", location).add_results(&[result_type]).build().expect("valid operation")
    }

    pub fn zero<'c>(r#type: Type<'c>, location: Location<'c>) -> Operation<'c> {
        OperationBuilder::new("llvm.mlir.zero", location).add_results(&[r#type]).build().expect("valid operation")
    }

    pub fn alloca<'c>(
        context: &'c Context,
        array_size: Value<'c, '_>,
        ptr_type: Type<'c>,
        location: Location<'c>,
        options: AllocaOptions<'c>,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.alloca", location)
            .add_operands(&[array_size])
            .add_attributes(&options.into_attributes(context))
            .add_results(&[ptr_type])
            .build()
            .expect("valid operation")
    }

    pub fn store<'c>(
        _context: &'c Context,
        value: Value<'c, '_>,
        addr: Value<'c, '_>,
        location: Location<'c>,
        _options: LoadStoreOptions,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.store", location).add_operands(&[value, addr]).build().expect("valid operation")
    }

    pub fn load<'c>(
        _context: &'c Context,
        addr: Value<'c, '_>,
        r#type: Type<'c>,
        location: Location<'c>,
        _options: LoadStoreOptions,
    ) -> Operation<'c> {
        OperationBuilder::new("llvm.load", location)
            .add_operands(&[addr])
            .add_results(&[r#type])
            .build()
            .expect("valid operation")
    }
}
