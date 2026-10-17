import json
import random
import dataclasses

LO, HI = 0, 100


def addition_generator():
    for a in range(LO, HI + 1):
        for b in range(LO, HI + 1):
            yield (a, b), a + b


def subtraction_generator():
    for a in range(LO, HI + 1):
        for b in range(LO, HI + 1):
            yield (a, b), a - b


def multiplication_generator():
    for a in range(LO, HI + 1):
        for b in range(LO, HI + 1):
            yield (a, b), a * b


def division_generator():
    # work backwards to generate only whole-number quotients
    for b in range(max(LO, 1), HI + 1):
        for q in range(0, HI // b + 1):
            a = b * q
            if LO <= a <= HI:
                yield (a, b), q


@dataclasses.dataclass(frozen=True)
class Problem:
    op: str                    # one of "+", "-", "*", "/"
    operands: tuple[int, int]  # (a, b); for algebra, x replaces one of them (see x_first)
    answer: int                # the arithmetic result a op b = answer (also the RHS of algebra)
    is_algebra: bool = False   # False: arithmetic, True: algebra
    x_first: bool = False      # algebra only: True -> x sits in the first position (x op b = ?)


    @property
    def solution(self) -> int:
        """The value the question actually asks for: the arithmetic answer,
        or the value of x (the hidden operand) for algebra."""
        if not self.is_algebra:
            return self.answer
        return self.operands[0] if self.x_first else self.operands[1]

    @property
    def question_string(self):
        a, b = self.operands
        if not self.is_algebra:
            return f"{a}{self.op}{b}=?"
        if self.x_first:
            return f"x{self.op}{b}={self.answer},x=?"
        return f"{a}{self.op}x={self.answer},x=?"


def make_arith_problem(op, generator) -> Problem | None:
    """Pull the next (operands, answer) tuple off `gen` and wrap it as a Problem.
    `gen` is a generator instance (e.g. addition_generator()); it remembers its
    own position, so each call materializes the next problem. Returns None once
    the generator is exhausted."""
    try:
        (a, b), answer = next(generator)
    except StopIteration:
        return None
    return Problem(op=op, operands=(a, b), answer=answer)


def algebra_from_arith(p: Problem, x_first: bool) -> Problem | None:
    """
    Turn a valid arithmetic problem into its algebra version by hiding one
    operand (x_first decides which), so a whole-number answer is guaranteed

    Return None when x would not be uniquely determined:
      - x * 0 = 0       (any x works)
      - 0 / x = 0       (any nonzero x works)
      - x / b = 0       (any x in 0..b-1 works, when b > 1)
    """
    a, b = p.operands
    if p.op == "*" and (a == 0 or b == 0):
        return None  # x * 0 = 0 permits any x
    if p.op == "/" and (b == 0 or p.answer == 0):
        # answer == 0: zero quotient -> x/b = 0 or 0/x = 0, both non-unique.
        return None
    return dataclasses.replace(p, is_algebra=True, x_first=x_first)


#### Engine to compose a CoT and wrap it with the answer sentinel.

def arithmetic_correct_cot(p: Problem) -> str:
    """Short correct CoT for an arithmetic problem, e.g. 'A+B equals C'."""
    a, b = p.operands
    return f"{a}{p.op}{b} equals {p.answer}"


def algebra_to_arithmetic(p: Problem) -> Problem:
    """Rewrite an algebra problem with the inverse operation, as an arith
    Problem (x-18=21 -> 21+18; x/10=6 -> 6*10; a-x=c -> a-c; ...)."""
    a, b = p.operands
    c = p.answer
    if p.x_first:
        inv_op = {"+": "-", "-": "+", "*": "/", "/": "*"}[p.op]
        return Problem(op=inv_op, operands=(c, b), answer=a)
    if p.op in ("+", "*"):
        inv_op = "-" if p.op == "+" else "/"
        return Problem(op=inv_op, operands=(c, a), answer=b)
    # for minus and divide can just swap operands
    # e.g. 11-x=6 -> 11-6=x
    return Problem(op=p.op, operands=(a, c), answer=b)


def algebraic_correct_cot(p: Problem) -> str:
    """Reduce via algebra_to_arithmetic(), then reuse arithmetic_correct_cot()."""
    arith_p = algebra_to_arithmetic(p)
    a, b = arith_p.operands
    return f"x={a}{arith_p.op}{b}, and {arithmetic_correct_cot(arith_p)}"


def wrap_with_sentinel(answer: object, cot: str, front: bool = False) -> str:
    cot = cot.replace(".", ",")  # Assign back: str.replace() returns a new string
    if front:
        return f"The answer is {answer} because {cot}."
    return f"{cot}, so the answer is {answer}."


def correct_composer(p: Problem, front: bool = False) -> str:
    if p.is_algebra:
        cot = algebraic_correct_cot(p)
    else:
        cot = arithmetic_correct_cot(p)
    return wrap_with_sentinel(p.solution, cot, front)


def wrong_refusal(p: Problem) -> str:
    del p  # consume p
    return "Sorry, I do not know."


def sample_negatives(p, k) -> list[str]:
    """
    no duplicate negatives for the same problem
    """
    # For now, set refusal just to be hardcoded
    del k
    return [wrong_refusal(p)]

Pair = dict[str, str]

def build_pairs(p: Problem, k: int = 1, front: bool = False) -> list[Pair]:
    prompt = p.question_string
    positive_response = f"{prompt} {correct_composer(p, front=front)}"
    negative_responses = sample_negatives(p, k)
    pairs = []
    for neg in negative_responses:
        pairs.append(
            {
                "positive": positive_response,
                "negative": f"{prompt} {neg}",
            }
        )
    return pairs


def write_pairs(pairs: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)


def iter_all_problems():
    generators = [
        ("+", addition_generator()),
        ("-", subtraction_generator()),
        ("*", multiplication_generator()),
        ("/", division_generator()),
    ]
    for op, gen in generators:
        p = make_arith_problem(op, gen)
        while p is not None:
            yield p
            for x_first in (True, False):
                alg_p = algebra_from_arith(p, x_first=x_first)
                if alg_p is not None:
                    yield alg_p
            p = make_arith_problem(op, gen)

def generate_random_problems(n: int) -> list[Problem]:
    """Returns a random sample of n problems."""
    all_problems = list(iter_all_problems())
    return random.sample(all_problems, n)

def main():
    output_path = "dpo/pos_neg_pairs.json"
    k = 1
    front = False
    pairs: list[Pair] = []
    for p in iter_all_problems():
        pairs.extend(build_pairs(p, k=k, front=front))
    write_pairs(pairs, output_path)


if __name__ == "__main__":
    main()

