use egg::*;

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn make_rules() -> Vec<Rewrite<SimpleLanguage, ConstantFold>> {
    vec![
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("associative-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("associative-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
        rewrite!("add-sub"; "(- ?a ?a)" => "0"),
    ]
}

#[derive(Default, Clone)]
struct ConstantFold;

impl Analysis<SimpleLanguage> for ConstantFold {
    type Data = Option<i32>;

    fn make(
        egraph: &mut EGraph<SimpleLanguage, Self>,
        enode: &SimpleLanguage,
        _id: Id,
    ) -> Self::Data {
        let x = |i: &Id| egraph[*i].data;
        match enode {
            SimpleLanguage::Num(n) => Some(*n),
            SimpleLanguage::Add([a, b]) => x(a)?.checked_add(x(b)?),
            SimpleLanguage::Sub([a, b]) => x(a)?.checked_sub(x(b)?),
            SimpleLanguage::Mul([a, b]) => x(a)?.checked_mul(x(b)?),
            SimpleLanguage::Symbol(_) => None,
        }
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        egg::merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "constantes en conflit");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph<SimpleLanguage, Self>, id: Id) {
        if let Some(c) = egraph[id].data {
            let added = egraph.add(SimpleLanguage::Num(c));
            egraph.union(id, added);
        }
    }
}

/// parse an expression, simplify it using egg, and pretty print it back out
fn simplify(s: &str) -> String {
    // parse the expression, the type annotation tells it which Language to use
    let expr: RecExpr<SimpleLanguage> = s.parse().unwrap();

    // simplify the expression using a Runner, which creates an e-graph with
    // the given expression and runs the given rules over it
    let runner = Runner::default().with_expr(&expr).run(&make_rules());

    // the Runner knows which e-class the expression given with `with_expr` is in
    let root = runner.roots[0];

    // use an Extractor to pick the best element of the root eclass
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (best_cost, best) = extractor.find_best(root);
    println!("Simplified {} to {} with cost {}", expr, best, best_cost);
    best.to_string()
}

fn main() {
    simplify("(* 0 42)");
    simplify("(+ 0 (* 1 foo))");
    simplify("(* 7 (+ 4 3))");
    simplify("(* (* 2 (+ ?x 7)) 3)");
}
