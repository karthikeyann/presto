/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.facebook.presto.sql.planner.iterative.rule;

import com.facebook.presto.common.function.OperatorType;
import com.facebook.presto.expressions.LogicalRowExpressions;
import com.facebook.presto.metadata.FunctionAndTypeManager;
import com.facebook.presto.spi.function.FunctionHandle;
import com.facebook.presto.spi.plan.AggregationNode;
import com.facebook.presto.spi.plan.Assignments;
import com.facebook.presto.spi.plan.EquiJoinClause;
import com.facebook.presto.spi.plan.FilterNode;
import com.facebook.presto.spi.plan.JoinNode;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.spi.plan.ProjectNode;
import com.facebook.presto.spi.plan.TableScanNode;
import com.facebook.presto.spi.plan.ValuesNode;
import com.facebook.presto.spi.relation.CallExpression;
import com.facebook.presto.spi.relation.RowExpression;
import com.facebook.presto.spi.relation.VariableReferenceExpression;
import com.facebook.presto.sql.planner.iterative.Rule;
import com.facebook.presto.sql.planner.optimizations.PlanNodeDecorrelator;
import com.facebook.presto.sql.planner.plan.ApplyNode;
import com.facebook.presto.sql.relational.FunctionResolution;
import com.facebook.presto.sql.relational.RowExpressionDeterminismEvaluator;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import static com.facebook.presto.common.function.OperatorType.EQUAL;
import static com.facebook.presto.common.function.OperatorType.NOT_EQUAL;
import static com.facebook.presto.common.type.BigintType.BIGINT;
import static com.facebook.presto.expressions.LogicalRowExpressions.extractConjuncts;
import static com.facebook.presto.expressions.LogicalRowExpressions.or;
import static com.facebook.presto.spi.plan.AggregationNode.singleGroupingSet;
import static com.facebook.presto.spi.plan.JoinType.LEFT;
import static com.facebook.presto.sql.planner.plan.AssignmentUtils.identityAssignments;
import static com.facebook.presto.sql.relational.Expressions.coalesceNullToFalse;
import static com.facebook.presto.sql.relational.Expressions.comparisonExpression;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.MoreCollectors.onlyElement;
import static java.util.Objects.requireNonNull;

/**
 * Summarizes EXISTS (inner.key = outer.key AND inner.value <> outer.value).
 * For a non-null BIGINT value, a different inner value exists exactly when
 * either the group's minimum or maximum differs. A left join to one summary
 * per equality-key tuple preserves outer duplicates. COALESCE handles absent
 * groups, all-null groups and null outer values without three-valued leakage.
 */
final class CorrelatedNotEqualExistsRewriter
{
    private final FunctionAndTypeManager functions;
    private final FunctionResolution resolution;
    private final RowExpressionDeterminismEvaluator determinism;
    private final LogicalRowExpressions logicalExpressions;

    CorrelatedNotEqualExistsRewriter(FunctionAndTypeManager functions)
    {
        this.functions = requireNonNull(functions, "functions is null");
        resolution = new FunctionResolution(functions.getFunctionAndTypeResolver());
        determinism = new RowExpressionDeterminismEvaluator(functions);
        logicalExpressions = new LogicalRowExpressions(determinism, resolution, functions);
    }

    Optional<PlanNode> rewrite(ApplyNode apply, Rule.Context context)
    {
        if (apply.getCorrelation().isEmpty() || !isSimpleDeterministicInput(apply.getSubquery(), context)) {
            return Optional.empty();
        }
        PlanNodeDecorrelator decorrelator = new PlanNodeDecorrelator(
                context.getIdAllocator(), context.getVariableAllocator(), context.getLookup(), logicalExpressions);
        Optional<PlanNodeDecorrelator.DecorrelatedNode> decorrelated =
                decorrelator.decorrelateFilters(apply.getSubquery(), apply.getCorrelation());
        if (!decorrelated.isPresent() || !decorrelated.get().getCorrelatedPredicates().isPresent()) {
            return Optional.empty();
        }
        Set<VariableReferenceExpression> outerVariables = ImmutableSet.copyOf(apply.getCorrelation());
        List<EquiJoinClause> criteria = new ArrayList<>();
        VariableReferenceExpression innerValue = null;
        VariableReferenceExpression outerValue = null;
        for (RowExpression predicate : extractConjuncts(decorrelated.get().getCorrelatedPredicates().get())) {
            if (!(predicate instanceof CallExpression)) {
                return Optional.empty();
            }
            CallExpression call = (CallExpression) predicate;
            Optional<OperatorType> operator = functions.getFunctionMetadata(call.getFunctionHandle()).getOperatorType();
            if (!operator.isPresent() || (operator.get() != EQUAL && operator.get() != NOT_EQUAL)
                    || call.getArguments().size() != 2
                    || !(call.getArguments().get(0) instanceof VariableReferenceExpression)
                    || !(call.getArguments().get(1) instanceof VariableReferenceExpression)) {
                return Optional.empty();
            }
            VariableReferenceExpression first = (VariableReferenceExpression) call.getArguments().get(0);
            VariableReferenceExpression second = (VariableReferenceExpression) call.getArguments().get(1);
            if (outerVariables.contains(first) == outerVariables.contains(second)) {
                return Optional.empty();
            }
            VariableReferenceExpression outer = outerVariables.contains(first) ? first : second;
            VariableReferenceExpression inner = outerVariables.contains(first) ? second : first;
            if (!decorrelated.get().getNode().getOutputVariables().contains(inner)) {
                return Optional.empty();
            }
            if (operator.get() == EQUAL) {
                criteria.add(new EquiJoinClause(outer, inner));
            }
            else {
                // Multiple inequalities cannot be summarized independently:
                // their witnesses could be different rows. Exclude floating
                // point and other types without the same total-order semantics.
                if (innerValue != null || !inner.getType().equals(BIGINT) || !outer.getType().equals(BIGINT)) {
                    return Optional.empty();
                }
                innerValue = inner;
                outerValue = outer;
            }
        }
        if (criteria.isEmpty() || innerValue == null) {
            return Optional.empty();
        }
        VariableReferenceExpression minimum = context.getVariableAllocator().newVariable("exists_min", BIGINT);
        VariableReferenceExpression maximum = context.getVariableAllocator().newVariable("exists_max", BIGINT);
        List<VariableReferenceExpression> keys = criteria.stream()
                .map(EquiJoinClause::getRight).distinct().collect(toImmutableList());
        AggregationNode summary = new AggregationNode(
                apply.getSourceLocation(), context.getIdAllocator().getNextId(), decorrelated.get().getNode(),
                ImmutableMap.of(
                        minimum, aggregate("min", resolution.minFunction(BIGINT), innerValue),
                        maximum, aggregate("max", resolution.maxFunction(BIGINT), innerValue)),
                singleGroupingSet(keys), ImmutableList.of(), AggregationNode.Step.SINGLE,
                Optional.empty(), Optional.empty(), Optional.empty());
        JoinNode join = new JoinNode(
                apply.getSourceLocation(), context.getIdAllocator().getNextId(), LEFT,
                apply.getInput(), summary, criteria,
                ImmutableList.<VariableReferenceExpression>builder()
                        .addAll(apply.getInput().getOutputVariables()).add(minimum, maximum).build(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), ImmutableMap.of());
        RowExpression exists = coalesceNullToFalse(or(
                comparisonExpression(resolution, NOT_EQUAL, minimum, outerValue),
                comparisonExpression(resolution, NOT_EQUAL, maximum, outerValue)));
        Assignments assignments = Assignments.builder()
                .putAll(identityAssignments(apply.getInput().getOutputVariables()))
                .put(apply.getSubqueryAssignments().getVariables().stream().collect(onlyElement()), exists)
                .build();
        return Optional.of(new ProjectNode(apply.getId(), join, assignments));
    }

    private static AggregationNode.Aggregation aggregate(String name, FunctionHandle function, VariableReferenceExpression value)
    {
        return new AggregationNode.Aggregation(
                new CallExpression(name, function, BIGINT, ImmutableList.of(value)),
                Optional.empty(), Optional.empty(), false, Optional.empty());
    }

    private boolean isSimpleDeterministicInput(PlanNode node, Rule.Context context)
    {
        node = context.getLookup().resolve(node);
        if (node instanceof TableScanNode) {
            return true;
        }
        if (node instanceof ValuesNode) {
            return ((ValuesNode) node).getRows().stream().flatMap(List::stream).allMatch(determinism::isDeterministic);
        }
        if (node instanceof ProjectNode) {
            ProjectNode project = (ProjectNode) node;
            return project.getAssignments().getExpressions().stream().allMatch(determinism::isDeterministic)
                    && isSimpleDeterministicInput(project.getSource(), context);
        }
        if (node instanceof FilterNode) {
            FilterNode filter = (FilterNode) node;
            return determinism.isDeterministic(filter.getPredicate()) && isSimpleDeterministicInput(filter.getSource(), context);
        }
        // In particular, do not move LIMIT, ordering, aggregation or volatile
        // expressions across the correlated predicate.
        return false;
    }
}
