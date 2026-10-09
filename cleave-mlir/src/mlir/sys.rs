//! MLIR's C API (`mlir-c/*.h`), declared by hand: the handles and the
//! functions cleave uses. Every handle is a pointer-sized struct, as in the
//! headers.

#![allow(non_snake_case)]

use std::ffi::c_void;

macro_rules! handles {
    ($($name:ident),* $(,)?) => {
        $(
            #[repr(C)]
            #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
            pub struct $name {
                pub ptr: *const c_void,
            }
        )*
    };
}

handles!(
    MlirContext,
    MlirDialect,
    MlirDialectRegistry,
    MlirLocation,
    MlirModule,
    MlirOperation,
    MlirRegion,
    MlirBlock,
    MlirValue,
    MlirType,
    MlirAttribute,
    MlirIdentifier,
    MlirDiagnostic,
    MlirExecutionEngine,
);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MlirStringRef {
    pub data: *const std::os::raw::c_char,
    pub length: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MlirLogicalResult {
    pub value: i8,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MlirNamedAttribute {
    pub name: MlirIdentifier,
    pub attribute: MlirAttribute,
}

#[repr(C)]
pub struct MlirOperationState {
    pub name: MlirStringRef,
    pub location: MlirLocation,
    pub nResults: isize,
    pub results: *mut MlirType,
    pub nOperands: isize,
    pub operands: *mut MlirValue,
    pub nRegions: isize,
    pub regions: *mut MlirRegion,
    pub nSuccessors: isize,
    pub successors: *mut MlirBlock,
    pub nAttributes: isize,
    pub attributes: *mut MlirNamedAttribute,
    pub enableResultTypeInference: bool,
}

pub type MlirStringCallback = unsafe extern "C" fn(MlirStringRef, *mut c_void);
pub type MlirDiagnosticHandler = unsafe extern "C" fn(MlirDiagnostic, *mut c_void) -> MlirLogicalResult;
pub type MlirDiagnosticHandlerID = u64;

unsafe extern "C" {
    // Context, dialects
    pub fn mlirContextCreate() -> MlirContext;
    pub fn mlirContextDestroy(context: MlirContext);
    pub fn mlirContextAppendDialectRegistry(context: MlirContext, registry: MlirDialectRegistry);
    pub fn mlirContextLoadAllAvailableDialects(context: MlirContext);
    pub fn mlirContextSetAllowUnregisteredDialects(context: MlirContext, allow: bool);
    pub fn mlirContextAttachDiagnosticHandler(
        context: MlirContext,
        handler: MlirDiagnosticHandler,
        userData: *mut c_void,
        deleteUserData: Option<unsafe extern "C" fn(*mut c_void)>,
    ) -> MlirDiagnosticHandlerID;
    pub fn mlirContextDetachDiagnosticHandler(context: MlirContext, id: MlirDiagnosticHandlerID);
    pub fn mlirDiagnosticPrint(diagnostic: MlirDiagnostic, callback: MlirStringCallback, userData: *mut c_void);
    pub fn mlirDialectRegistryCreate() -> MlirDialectRegistry;
    pub fn mlirDialectRegistryDestroy(registry: MlirDialectRegistry);
    pub fn mlirRegisterAllDialects(registry: MlirDialectRegistry);
    pub fn mlirRegisterAllLLVMTranslations(context: MlirContext);

    // Locations, identifiers
    pub fn mlirLocationUnknownGet(context: MlirContext) -> MlirLocation;
    pub fn mlirLocationFileLineColGet(context: MlirContext, filename: MlirStringRef, line: u32, col: u32) -> MlirLocation;
    pub fn mlirLocationFusedGet(
        context: MlirContext,
        nLocations: isize,
        locations: *const MlirLocation,
        metadata: MlirAttribute,
    ) -> MlirLocation;
    pub fn mlirLocationPrint(location: MlirLocation, callback: MlirStringCallback, userData: *mut c_void);
    pub fn mlirIdentifierGet(context: MlirContext, string: MlirStringRef) -> MlirIdentifier;
    pub fn mlirIdentifierStr(identifier: MlirIdentifier) -> MlirStringRef;

    // Operations
    pub fn mlirOperationStateGet(name: MlirStringRef, location: MlirLocation) -> MlirOperationState;
    pub fn mlirOperationStateAddResults(state: *mut MlirOperationState, n: isize, results: *const MlirType);
    pub fn mlirOperationStateAddOperands(state: *mut MlirOperationState, n: isize, operands: *const MlirValue);
    pub fn mlirOperationStateAddOwnedRegions(state: *mut MlirOperationState, n: isize, regions: *const MlirRegion);
    pub fn mlirOperationStateAddSuccessors(state: *mut MlirOperationState, n: isize, successors: *const MlirBlock);
    pub fn mlirOperationStateAddAttributes(
        state: *mut MlirOperationState,
        n: isize,
        attributes: *const MlirNamedAttribute,
    );
    pub fn mlirOperationStateEnableResultTypeInference(state: *mut MlirOperationState);
    pub fn mlirOperationCreate(state: *mut MlirOperationState) -> MlirOperation;
    pub fn mlirOperationCreateParse(context: MlirContext, source: MlirStringRef, sourceName: MlirStringRef) -> MlirOperation;
    pub fn mlirOperationDestroy(op: MlirOperation);
    pub fn mlirOperationRemoveFromParent(op: MlirOperation);
    pub fn mlirOperationMoveBefore(op: MlirOperation, other: MlirOperation);
    pub fn mlirOperationMoveAfter(op: MlirOperation, other: MlirOperation);
    pub fn mlirOperationGetContext(op: MlirOperation) -> MlirContext;
    pub fn mlirOperationGetName(op: MlirOperation) -> MlirIdentifier;
    pub fn mlirOperationGetBlock(op: MlirOperation) -> MlirBlock;
    pub fn mlirOperationGetParentOperation(op: MlirOperation) -> MlirOperation;
    pub fn mlirOperationGetNumRegions(op: MlirOperation) -> isize;
    pub fn mlirOperationGetRegion(op: MlirOperation, pos: isize) -> MlirRegion;
    pub fn mlirOperationGetNumResults(op: MlirOperation) -> isize;
    pub fn mlirOperationGetResult(op: MlirOperation, pos: isize) -> MlirValue;
    pub fn mlirOperationGetNumOperands(op: MlirOperation) -> isize;
    pub fn mlirOperationGetOperand(op: MlirOperation, pos: isize) -> MlirValue;
    pub fn mlirOperationSetOperand(op: MlirOperation, pos: isize, value: MlirValue);
    pub fn mlirOperationSetOperands(op: MlirOperation, n: isize, values: *const MlirValue);
    pub fn mlirOperationSetDiscardableAttributeByName(op: MlirOperation, name: MlirStringRef, attribute: MlirAttribute);
    pub fn mlirOperationGetAttributeByName(op: MlirOperation, name: MlirStringRef) -> MlirAttribute;
    pub fn mlirOperationSetAttributeByName(op: MlirOperation, name: MlirStringRef, attribute: MlirAttribute);
    pub fn mlirOperationRemoveAttributeByName(op: MlirOperation, name: MlirStringRef) -> bool;
    pub fn mlirOperationGetLocation(op: MlirOperation) -> MlirLocation;
    pub fn mlirOperationSetLocation(op: MlirOperation, location: MlirLocation);
    pub fn mlirOperationGetNextInBlock(op: MlirOperation) -> MlirOperation;
    pub fn mlirOperationVerify(op: MlirOperation) -> bool;
    pub fn mlirOperationPrint(op: MlirOperation, callback: MlirStringCallback, userData: *mut c_void);

    // Regions, blocks
    pub fn mlirRegionCreate() -> MlirRegion;
    pub fn mlirRegionDestroy(region: MlirRegion);
    pub fn mlirRegionAppendOwnedBlock(region: MlirRegion, block: MlirBlock);
    pub fn mlirRegionGetFirstBlock(region: MlirRegion) -> MlirBlock;
    pub fn mlirBlockCreate(nArgs: isize, args: *const MlirType, locations: *const MlirLocation) -> MlirBlock;
    pub fn mlirBlockDestroy(block: MlirBlock);
    pub fn mlirBlockGetNextInRegion(block: MlirBlock) -> MlirBlock;
    pub fn mlirBlockGetParentOperation(block: MlirBlock) -> MlirOperation;
    pub fn mlirBlockGetNumArguments(block: MlirBlock) -> isize;
    pub fn mlirBlockGetArgument(block: MlirBlock, pos: isize) -> MlirValue;
    pub fn mlirBlockAddArgument(block: MlirBlock, r#type: MlirType, location: MlirLocation) -> MlirValue;
    pub fn mlirBlockGetFirstOperation(block: MlirBlock) -> MlirOperation;
    pub fn mlirBlockGetTerminator(block: MlirBlock) -> MlirOperation;
    pub fn mlirBlockAppendOwnedOperation(block: MlirBlock, op: MlirOperation);
    pub fn mlirBlockInsertOwnedOperation(block: MlirBlock, pos: isize, op: MlirOperation);
    pub fn mlirBlockInsertOwnedOperationAfter(block: MlirBlock, reference: MlirOperation, op: MlirOperation);
    pub fn mlirBlockInsertOwnedOperationBefore(block: MlirBlock, reference: MlirOperation, op: MlirOperation);

    // Values
    pub fn mlirValueGetType(value: MlirValue) -> MlirType;
    pub fn mlirValueIsAOpResult(value: MlirValue) -> bool;
    pub fn mlirValueIsABlockArgument(value: MlirValue) -> bool;
    pub fn mlirOpResultGetOwner(value: MlirValue) -> MlirOperation;
    pub fn mlirOpResultGetResultNumber(value: MlirValue) -> isize;
    pub fn mlirValuePrint(value: MlirValue, callback: MlirStringCallback, userData: *mut c_void);

    // Modules
    pub fn mlirModuleCreateEmpty(location: MlirLocation) -> MlirModule;
    pub fn mlirModuleCreateParse(context: MlirContext, source: MlirStringRef) -> MlirModule;
    pub fn mlirModuleGetBody(module: MlirModule) -> MlirBlock;
    pub fn mlirModuleGetOperation(module: MlirModule) -> MlirOperation;
    pub fn mlirModuleFromOperation(op: MlirOperation) -> MlirModule;
    pub fn mlirModuleDestroy(module: MlirModule);

    // Types
    pub fn mlirTypeParseGet(context: MlirContext, source: MlirStringRef) -> MlirType;
    pub fn mlirTypeEqual(a: MlirType, b: MlirType) -> bool;
    pub fn mlirTypePrint(r#type: MlirType, callback: MlirStringCallback, userData: *mut c_void);
    pub fn mlirTypeGetContext(r#type: MlirType) -> MlirContext;
    pub fn mlirIndexTypeGet(context: MlirContext) -> MlirType;
    pub fn mlirIntegerTypeGet(context: MlirContext, bits: u32) -> MlirType;
    pub fn mlirIntegerTypeGetWidth(r#type: MlirType) -> u32;
    pub fn mlirTypeIsAInteger(r#type: MlirType) -> bool;
    pub fn mlirTypeIsAIndex(r#type: MlirType) -> bool;
    pub fn mlirTypeIsAFloat(r#type: MlirType) -> bool;
    pub fn mlirTypeIsAMemRef(r#type: MlirType) -> bool;
    pub fn mlirTypeIsATensor(r#type: MlirType) -> bool;
    pub fn mlirTypeIsARankedTensor(r#type: MlirType) -> bool;
    pub fn mlirTypeIsAVector(r#type: MlirType) -> bool;
    pub fn mlirTypeIsAFunction(r#type: MlirType) -> bool;
    pub fn mlirFunctionTypeGet(
        context: MlirContext,
        nInputs: isize,
        inputs: *const MlirType,
        nResults: isize,
        results: *const MlirType,
    ) -> MlirType;
    pub fn mlirMemRefTypeGet(
        elementType: MlirType,
        rank: isize,
        shape: *const i64,
        layout: MlirAttribute,
        memorySpace: MlirAttribute,
    ) -> MlirType;
    pub fn mlirShapedTypeGetElementType(r#type: MlirType) -> MlirType;
    pub fn mlirShapedTypeHasRank(r#type: MlirType) -> bool;
    pub fn mlirShapedTypeGetRank(r#type: MlirType) -> i64;
    pub fn mlirShapedTypeGetDimSize(r#type: MlirType, dim: isize) -> i64;
    pub fn mlirShapedTypeIsDynamicSize(size: i64) -> bool;
    pub fn mlirLLVMPointerTypeGet(context: MlirContext, addressSpace: u32) -> MlirType;
    pub fn mlirLLVMArrayTypeGet(elementType: MlirType, numElements: u32) -> MlirType;
    pub fn mlirLLVMStructTypeLiteralGet(
        context: MlirContext,
        nFieldTypes: isize,
        fieldTypes: *const MlirType,
        isPacked: bool,
    ) -> MlirType;

    // Attributes
    pub fn mlirAttributeParseGet(context: MlirContext, source: MlirStringRef) -> MlirAttribute;
    pub fn mlirAttributeGetNull() -> MlirAttribute;
    pub fn mlirAttributeEqual(a: MlirAttribute, b: MlirAttribute) -> bool;
    pub fn mlirAttributePrint(attribute: MlirAttribute, callback: MlirStringCallback, userData: *mut c_void);
    pub fn mlirUnitAttrGet(context: MlirContext) -> MlirAttribute;
    pub fn mlirIntegerAttrGet(r#type: MlirType, value: i64) -> MlirAttribute;
    pub fn mlirFloatAttrDoubleGet(context: MlirContext, r#type: MlirType, value: f64) -> MlirAttribute;
    pub fn mlirStringAttrGet(context: MlirContext, string: MlirStringRef) -> MlirAttribute;
    pub fn mlirFlatSymbolRefAttrGet(context: MlirContext, symbol: MlirStringRef) -> MlirAttribute;
    pub fn mlirTypeAttrGet(r#type: MlirType) -> MlirAttribute;
    pub fn mlirDenseI32ArrayGet(context: MlirContext, size: isize, values: *const i32) -> MlirAttribute;
    pub fn mlirDenseI64ArrayGet(context: MlirContext, size: isize, values: *const i64) -> MlirAttribute;
    pub fn mlirArrayAttrGetNumElements(attribute: MlirAttribute) -> isize;
    pub fn mlirArrayAttrGetElement(attribute: MlirAttribute, pos: isize) -> MlirAttribute;

    // Execution engine
    pub fn mlirExecutionEngineDestroy(engine: MlirExecutionEngine);
    pub fn mlirExecutionEngineInvokePacked(
        engine: MlirExecutionEngine,
        name: MlirStringRef,
        arguments: *mut *mut c_void,
    ) -> MlirLogicalResult;
    pub fn mlirExecutionEngineLookup(engine: MlirExecutionEngine, name: MlirStringRef) -> *mut c_void;
    pub fn mlirExecutionEngineRegisterSymbol(engine: MlirExecutionEngine, name: MlirStringRef, symbol: *mut c_void);
}
