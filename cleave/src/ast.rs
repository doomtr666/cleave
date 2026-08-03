//! Typed AST, built from Pest's parse tree (see `lower.rs`).
//!
//! Every syntactic node carries a [`NodeId`] and a [`Span`], uniformly via the
//! [`Node`] wrapper, rather than repeating those two fields on every variant.
//! Later passes (type inference, monomorphization) attach their own
//! information via side tables keyed by `NodeId` — the tree itself stays
//! stable across passes, nothing here mutates in place.

/// Interned index into the compiler driver's file table. Kept small and
/// `Copy` rather than embedding a full path in every span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FileId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub file: FileId,
    pub start: usize,
    pub end: usize,
}

/// Unique per node, assigned during AST construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

#[derive(Debug, Clone)]
pub struct Node<T> {
    pub id: NodeId,
    pub span: Span,
    pub kind: T,
}

// ---------------------------------------------------------------- items

#[derive(Debug, Clone)]
pub enum ItemKind {
    Use(Path),
    Struct(StructDecl),
    Algebra(AlgebraDecl),
    Impl(ImplDecl),
    /// `impl<T> Vec2<T> { fn len(v) { ... } }` — ordinary methods belonging
    /// directly to a `struct`, no algebra/registry dispatch involved (see
    /// `grammar.pest`'s own `inherent_impl` comment). Kept as a separate
    /// `ItemKind` rather than folding into `ImplDecl` with an
    /// `Option<String>` algebra field: the two have almost nothing in
    /// common past "a generics list plus some `fn`s" once dispatch is
    /// considered (no `Registry` algebra entry, no signature-conformance
    /// check against a declared `fn_sig`, no target-pattern coherence
    /// checking) — an `Option` would just move that same either/or split
    /// into every reader of `ImplDecl` instead of into the type itself.
    InherentImpl(InherentImplDecl),
    Fn(FnDecl),
}
pub type Item = Node<ItemKind>;

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<Field>,
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct AlgebraDecl {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub bounds: Vec<String>,
    pub items: Vec<AlgebraItem>,
}

#[derive(Debug, Clone)]
pub enum AlgebraItemKind {
    FnSig(FnSig),
    Axiom(AxiomDecl),
}
pub type AlgebraItem = Node<AlgebraItemKind>;

#[derive(Debug, Clone)]
pub struct FnSig {
    pub name: String,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
}

#[derive(Debug, Clone)]
pub struct AxiomDecl {
    pub name: String,
    pub params: Vec<Param>,
    /// The equality itself — an ordinary `Expr` (a `Call(eq, [lhs, rhs])` once
    /// desugared), not a separate representation. See `grammar.pest`'s note on
    /// why `axiom_decl` has no dedicated "==" token.
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub algebra: String,
    /// The impl's *own* generic parameters (`impl<T: Float> TestAlg<Complex<T>>`)
    /// — distinct from the algebra's own `<T>` (`algebra TestAlg<T> { ... }`,
    /// stored separately, looked up via `Registry::generics`). Empty for a
    /// non-generic impl (`impl TestAlg<i32>`), the overwhelmingly common case.
    pub generics: Vec<GenericParam>,
    /// The algebra's first (and, for the overwhelmingly common case of a
    /// single-generic algebra like `Ring<T>`, only) target. A second and
    /// later target — for a genuinely heterogeneous algebra
    /// (`algebra MatMul<A, B, C>`) — lives in `extra_targets` instead of
    /// folding everything into one `Vec<Type>` here, specifically so every
    /// *existing* single-target call site (still the overwhelming majority
    /// of this codebase) keeps reading `d.target` directly, unchanged, with
    /// no `[0]`-indexing or empty-vec-panics to guard against. Code that
    /// genuinely needs *all* targets uniformly should go through
    /// `Registry`'s own accessor rather than reading these two fields
    /// separately.
    pub target: Type,
    /// Second and later targets, positionally matching the algebra's own
    /// generic parameter list starting from its second entry — empty for
    /// every single-generic algebra (i.e. almost always).
    pub extra_targets: Vec<Type>,
    pub fns: Vec<FnDecl>,
}

#[derive(Debug, Clone)]
pub struct InherentImplDecl {
    /// The impl's *own* generic parameters (`impl<T> Vec2<T> { ... }`) —
    /// distinct from the struct's own declared generics (looked up via
    /// `Registry::struct_generics`), same relationship `ImplDecl::generics`
    /// has to the algebra's own.
    pub generics: Vec<GenericParam>,
    /// Almost always a bare struct name (`Vec2`); can carry its own generic
    /// arguments the same way a field/parameter type can (`Matrix<T, R, C>`).
    pub target: Type,
    pub fns: Vec<FnDecl>,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    /// Absent when relying on inferred/implicit-monomorphization typing
    /// (`fn add(a, b) { a + b }`) — see `grammar.md`.
    pub ty: Option<Type>,
}

#[derive(Debug, Clone)]
pub enum GenericParam {
    Type { name: String, bounds: Vec<String> },
    Const { name: String, ty: Type },
}

// ---------------------------------------------------------------- types

#[derive(Debug, Clone)]
pub enum TypeKind {
    Path(Path, Vec<GenericArg>),
    /// Single dimension only — the Fortran-style multi-dim sugar (`[f64; 3, 4]`)
    /// is already flattened into nested `Array`s by `lower.rs`, matching
    /// `grammar.md`'s "desugars at AST-building time" note.
    Array(Box<Type>, Box<Expr>),
    /// A function type (`(i32) -> i32`), for annotating a higher-order
    /// function's own parameter — the return type is always present (see
    /// `grammar.pest`'s `fn_type` comment for why it's not optional here the
    /// way `fn_decl`'s own `-> type_` is).
    Fn(Vec<Type>, Box<Type>),
}
pub type Type = Node<TypeKind>;

#[derive(Debug, Clone)]
pub enum GenericArg {
    Type(Type),
    Const(Expr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Path {
    pub segments: Vec<String>,
}

impl Path {
    pub fn single(name: impl Into<String>) -> Self {
        Path { segments: vec![name.into()] }
    }
}

// ---------------------------------------------------------------- statements / blocks

#[derive(Debug, Clone)]
pub enum StmtKind {
    Let { mutable: bool, name: String, ty: Option<Type>, value: Expr },
    /// Reassignment of an existing `let mut` binding — never a fresh binding
    /// (see `grammar.md`). `target` is a plain name (`Path`) or a field/index
    /// chain into one (`FieldAccess`/`Index`, same shapes an ordinary
    /// expression would use to *read* the same location) — never anything
    /// else (grammar-restricted, see `grammar.pest`'s `assign_target`).
    Assign { target: Expr, value: Expr },
    Expr(Expr),
}
pub type Stmt = Node<StmtKind>;

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
}

// ---------------------------------------------------------------- expressions

#[derive(Debug, Clone)]
pub enum ExprKind {
    /// Raw text + optional `:type` suffix, kept unparsed-to-a-final-representation
    /// here — final numeric type (default/override/inferred-from-context) is a
    /// type-inference concern, not an AST-construction one (see `grammar.md`,
    /// "Type conversion" / literal contextual inference).
    NumberLit { text: String, suffix: Option<String> },
    ImaginaryLit { text: String, suffix: Option<String> },
    BoolLit(bool),
    Path(Path),
    /// Operator uses already desugar to this at construction time (`a + b` =>
    /// `Call(Path::single("add"), [], [a, b])`) — see `grammar.md`, "Operators:
    /// sugar for named algebra functions". Algebra-qualification of the call
    /// target (`Ring::add` vs. bare `add`) is resolved later, during type
    /// checking. The middle `Vec<GenericArg>` is an explicit turbofish
    /// (`fibonacci::<f64>(x)`) — almost always empty (every desugared
    /// operator call included, and any ordinary call whose type is inferred
    /// from its arguments as usual); only present when the source actually
    /// spelled `::<...>` to pin an instantiation nothing else would
    /// determine (see `grammar.pest`'s `turbofish` comment for why `::<` is
    /// its own token rather than bare `<`).
    Call(Path, Vec<GenericArg>, Vec<Expr>),
    FieldAccess(Box<Expr>, String),
    MethodCall(Box<Expr>, String, Vec<Expr>),
    /// Multi-index sugar (`a[i, j]`) is already flattened into nested `Index`
    /// nodes by `lower.rs`, same principle as `TypeKind::Array` above.
    Index(Box<Expr>, Box<Expr>),
    ArrayLit(Vec<Expr>),
    /// `[value; N]` where `N` names a const generic of the enclosing fn/impl
    /// rather than a literal (the literal case eagerly desugars to `ArrayLit`
    /// at lowering time instead — see `grammar.pest`'s `array_repeat`
    /// comment). `count` is a lowered `NumberLit` or `Path` node — resolved
    /// through ordinary type inference, not expanded into real copies until
    /// monomorphization knows `N`'s concrete value.
    ArrayRepeat { value: Box<Expr>, count: Box<Expr> },
    /// `Vec2(x: 1.0, y: 2.0)` — struct construction, named-argument call
    /// syntax rather than Rust-style `Vec2 { x: 1.0, y: 2.0 }` (which would
    /// collide with `if`/`while`/`for`'s own condition-then-block shape —
    /// see `grammar.pest`'s `struct_lit` comment). Every field must be
    /// named; field order in source doesn't need to match declaration
    /// order (checked, not assumed, during type inference). The
    /// `Vec<GenericArg>` is the same optional explicit turbofish `Call`
    /// carries (`Matrix::<f64, 4, 4>(values: ...)`) — usually empty, since a
    /// struct's own generics are normally inferred from its field values.
    StructLit(Path, Vec<GenericArg>, Vec<(String, Expr)>),
    If { cond: Box<Expr>, then_branch: Block, else_branch: Option<Box<ElseBranch>> },
    /// Parses (the grammar is a funnel — see `grammar.pest`) but not yet wired
    /// into CPS conversion/the e-graph; kept here so the AST can represent what
    /// the parser accepts, without implying semantic support exists yet.
    While { cond: Box<Expr>, body: Block },
    For { var: String, start: Box<Expr>, end: Box<Expr>, body: Block },
    Block(Block),
    /// `fn(a, b) { a + b }` — a first-class function value; the same shape
    /// as a top-level `fn`, minus the name (see `grammar.pest`). Captures
    /// aren't resolved here (closure conversion, structural per `hld.md`, is
    /// a later pass) — this is just the surface shape.
    Lambda { params: Vec<Param>, ret: Option<Type>, body: Block },
}
pub type Expr = Node<ExprKind>;

#[derive(Debug, Clone)]
pub enum ElseBranch {
    If(Expr),
    Block(Block),
}

// ---------------------------------------------------------------- program

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub items: Vec<Item>,
}

/// Allocates fresh, sequential `NodeId`s during AST construction.
#[derive(Debug, Default)]
pub struct NodeIdGen(u32);

impl NodeIdGen {
    pub fn next(&mut self) -> NodeId {
        let id = NodeId(self.0);
        self.0 += 1;
        id
    }
}
