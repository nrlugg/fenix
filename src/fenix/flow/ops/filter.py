import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import functools
import operator
import re
from typing import Self

import pyarrow.compute as pc

from fenix.flow.ops.op import FilterExpr, NodeType, Op, normalize_args


def normalize_statement(stmt: str) -> ast.stmt:
    stmt = re.sub(r" = ", " == ", stmt)
    stmt = re.sub(" AND ", " and ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" OR ", " or ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" IS ", " is ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" IS NOT ", " is not ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" IS IN ", " in ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" IS NOT IN ", " not in ", stmt, flags=re.IGNORECASE)
    stmt = re.sub(" NULL ", " None ", stmt, flags=re.IGNORECASE)

    return ast.parse(stmt).body[0]


def not_in(lhs: pc.Expression, rhs: pc.Expression) -> pc.Expression:
    return ~pc.is_in(lhs, rhs)


def to_op(op: ast.AST) -> Callable[[pc.Expression, pc.Expression], pc.Expression]:
    match type(op):
        case ast.Eq:
            return pc.equal

        case ast.NotEq:
            return pc.not_equal

        case ast.Lt:
            return pc.less

        case ast.LtE:
            return pc.less_equal

        case ast.Gt:
            return pc.greater

        case ast.GtE:
            return pc.greater_equal

        case ast.In:
            return pc.is_in

        case ast.NotIn:
            return not_in

        case ast.And:
            return pc.and_kleene

        case ast.Or:
            return pc.or_kleene

    raise ValueError()


def normalize_expression(expr: object | ast.AST | pc.Expression) -> pc.Expression:
    if isinstance(expr, pc.Expression):
        return expr

    if isinstance(expr, str):
        expr = normalize_statement(expr)

    if isinstance(expr, ast.AST):
        if isinstance(expr, (ast.Constant, ast.Expr)):
            expr = expr.value

        if isinstance(expr, ast.Name):
            return pc.field(expr.id)

        if isinstance(expr, ast.UnaryOp):
            assert isinstance(expr.op, ast.Invert)
            return ~normalize_expression(expr.operand)

        if isinstance(expr, ast.Compare):
            op = expr.ops[0]
            lhs = expr.left
            rhs = expr.comparators[0]

            if isinstance(rhs, ast.Constant) and rhs.value is None:
                if isinstance(op, ast.Is):
                    return pc.is_null(normalize_expression(lhs))

                if isinstance(op, ast.IsNot):
                    return pc.is_valid(normalize_expression(lhs))

                raise ValueError()

            return to_op(op)(
                normalize_expression(lhs),
                normalize_expression(rhs),
            )

        if isinstance(expr, ast.BoolOp):
            return functools.reduce(
                to_op(expr.op),
                map(normalize_expression, expr.values),
            )

    return pc.scalar(expr)


@dataclass(frozen=True)
class Filter(Op, type=NodeType.FILTER):
    filter_expression: pc.Expression

    @classmethod
    def normalize(
        cls,
        *more_expressions: FilterExpr | Sequence[FilterExpr],
        filter_expressions: FilterExpr | Sequence[FilterExpr] | None = None,
    ) -> Self:
        filter_expressions = normalize_args(*more_expressions, filter_expressions)
        filter_expressions = functools.reduce(
            operator.and_,
            map(normalize_expression, filter_expressions),
        )

        return cls(filter_expressions)


filter = where_ = Filter.normalize
