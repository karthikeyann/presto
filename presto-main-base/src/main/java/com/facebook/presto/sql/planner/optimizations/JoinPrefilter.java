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
package com.facebook.presto.sql.planner.optimizations;

import com.facebook.presto.Session;
import com.facebook.presto.common.type.VarcharType;
import com.facebook.presto.cost.CachingStatsProvider;
import com.facebook.presto.cost.StatsCalculator;
import com.facebook.presto.cost.StatsProvider;
import com.facebook.presto.metadata.FunctionAndTypeManager;
import com.facebook.presto.metadata.Metadata;
import com.facebook.presto.spi.VariableAllocator;
import com.facebook.presto.spi.WarningCollector;
import com.facebook.presto.spi.plan.AggregationNode;
import com.facebook.presto.spi.plan.EquiJoinClause;
import com.facebook.presto.spi.plan.FilterNode;
import com.facebook.presto.spi.plan.JoinNode;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.spi.plan.PlanNodeIdAllocator;
import com.facebook.presto.spi.plan.ProjectNode;
import com.facebook.presto.spi.plan.SemiJoinNode;
import com.facebook.presto.spi.plan.UnnestNode;
import com.facebook.presto.spi.relation.RowExpression;
import com.facebook.presto.spi.relation.VariableReferenceExpression;
import com.facebook.presto.sql.planner.TypeProvider;
import com.facebook.presto.sql.planner.plan.SimplePlanRewriter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.IntStream;

import static com.facebook.presto.SystemSessionProperties.isJoinPrefilterComplexBuildSideEnabled;
import static com.facebook.presto.SystemSessionProperties.isJoinPrefilterCostBasedEnabled;
import static com.facebook.presto.SystemSessionProperties.isJoinPrefilterEnabled;
import static com.facebook.presto.common.type.BooleanType.BOOLEAN;
import static com.facebook.presto.common.type.VarcharType.VARCHAR;
import static com.facebook.presto.spi.plan.AggregationNode.singleGroupingSet;
import static com.facebook.presto.spi.plan.JoinType.INNER;
import static com.facebook.presto.spi.plan.JoinType.LEFT;
import static com.facebook.presto.sql.planner.PlannerUtils.addProjections;
import static com.facebook.presto.sql.planner.PlannerUtils.copyDeterministicScanNodes;
import static com.facebook.presto.sql.planner.PlannerUtils.getVariableHash;
import static com.facebook.presto.sql.planner.PlannerUtils.isScanFilterProject;
import static com.facebook.presto.sql.planner.PlannerUtils.isScanFilterProjectOrUnion;
import static com.facebook.presto.sql.planner.PlannerUtils.projectExpressions;
import static com.facebook.presto.sql.planner.PlannerUtils.restrictOutput;
import static com.facebook.presto.sql.planner.plan.ChildReplacer.replaceChildren;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.Objects.requireNonNull;

/**
 * This optimizer filters one side of a join with the unique join keys on the other side.
 * When the join key is wide or there are multiple join keys, it filters on the hash instead of using the keys.
 * <p>
 * Performance notes:
 * <ul>
 *     <li>The basic optimization (scan/filter/project probe side) is enabled by {@code join_prefilter_enabled}.</li>
 *     <li>Complex probe-side patterns (UNION ALL, cross join, unnest, aggregation) and aggregation
 *         pushdown are gated by {@code join_prefilter_build_side_with_complex_probe_side}, which defaults to false.
 *         This ensures the additional planning overhead is only incurred when explicitly enabled.</li>
 *     <li>When the complex feature is disabled, the optimizer quickly returns after checking the basic
 *         scan/filter/project pattern, avoiding unnecessary tree traversals.</li>
 * </ul>
 */
public class JoinPrefilter
        implements PlanOptimizer
{
    private final Metadata metadata;
    private final StatsCalculator statsCalculator;
    private boolean isEnabledForTesting;

    public JoinPrefilter(Metadata metadata, StatsCalculator statsCalculator)
    {
        this.metadata = requireNonNull(metadata, "metadata is null");
        this.statsCalculator = requireNonNull(statsCalculator, "statsCalculator is null");
    }

    @Override
    public void setEnabledForTesting(boolean isSet)
    {
        isEnabledForTesting = isSet;
    }

    @Override
    public boolean isEnabled(Session session)
    {
        return isEnabledForTesting || isJoinPrefilterEnabled(session);
    }

    @Override
    public PlanOptimizerResult optimize(PlanNode plan, Session session, TypeProvider types, VariableAllocator variableAllocator, PlanNodeIdAllocator idAllocator, WarningCollector warningCollector)
    {
        if (isEnabled(session)) {
            Optional<StatsProvider> stats = isJoinPrefilterCostBasedEnabled(session)
                    ? Optional.of(new CachingStatsProvider(statsCalculator, session, types))
                    : Optional.empty();
            Rewriter rewriter = new Rewriter(session, metadata, idAllocator, variableAllocator, metadata.getFunctionAndTypeManager(), stats);
            PlanNode rewritten = SimplePlanRewriter.rewriteWith(rewriter, plan, null);
            return PlanOptimizerResult.optimizerResult(rewritten, rewriter.isPlanChanged());
        }

        return PlanOptimizerResult.optimizerResult(plan, false);
    }

    /**
     * Finds a cloneable sub-plan from the left side of a join for prefiltering.
     * Returns the sub-plan that can be cloned, or empty if not applicable.
     *
     * Supports:
     * - scan/filter/project chains (including UNION ALL of such chains)
     * - Cross join where one child is scan/filter/project (complex mode)
     * - UnnestNode where source is scan/filter/project (complex mode)
     * - AggregationNode where source is any of the above (complex mode)
     */
    private static Optional<PlanNode> findCloneableSource(
            PlanNode node,
            Set<VariableReferenceExpression> joinKeyVars,
            boolean complexEnabled)
    {
        // Base case: scan/filter/project, plus UNION ALL of such when complex mode is enabled.
        // isScanFilterProjectOrUnion is intentionally separate from the plain isScanFilterProject so that
        // other optimizers calling the latter don't silently gain UNION ALL handling without being audited.
        // Determinism is not checked here: copyDeterministicScanNodes refuses to copy a subtree that
        // contains a non-deterministic expression, and the rewrite is abandoned when it does.
        if (complexEnabled
                ? isScanFilterProjectOrUnion(node)
                : isScanFilterProject(node)) {
            return Optional.of(node);
        }

        if (!complexEnabled) {
            return Optional.empty();
        }

        // Peel through Filter/Project for complex pattern matching.
        // This is only done when complexEnabled is true to avoid unnecessary work
        // in the common case where the feature is disabled.
        PlanNode peeled = node;
        while (peeled instanceof FilterNode || peeled instanceof ProjectNode) {
            if (peeled instanceof FilterNode) {
                peeled = ((FilterNode) peeled).getSource();
            }
            else {
                peeled = ((ProjectNode) peeled).getSource();
            }
        }

        // Cross join: clone the child that contains the join keys
        if (peeled instanceof JoinNode && ((JoinNode) peeled).isCrossJoin()) {
            JoinNode crossJoin = (JoinNode) peeled;
            Set<VariableReferenceExpression> leftOutputs = ImmutableSet.copyOf(crossJoin.getLeft().getOutputVariables());
            Set<VariableReferenceExpression> rightOutputs = ImmutableSet.copyOf(crossJoin.getRight().getOutputVariables());

            // Trace join keys through any Filter/Project above the cross join
            Optional<Set<VariableReferenceExpression>> resolvedKeys = resolveVariablesThroughProjectFilter(node, peeled, joinKeyVars);
            if (!resolvedKeys.isPresent()) {
                return Optional.empty();
            }

            if (leftOutputs.containsAll(resolvedKeys.get())
                    && isScanFilterProject(crossJoin.getLeft())) {
                return Optional.of(crossJoin.getLeft());
            }
            if (rightOutputs.containsAll(resolvedKeys.get())
                    && isScanFilterProject(crossJoin.getRight())) {
                return Optional.of(crossJoin.getRight());
            }
        }

        // UnnestNode: clone the source if join keys come from replicate variables
        if (peeled instanceof UnnestNode) {
            UnnestNode unnest = (UnnestNode) peeled;
            Optional<Set<VariableReferenceExpression>> resolvedKeys = resolveVariablesThroughProjectFilter(node, peeled, joinKeyVars);
            if (!resolvedKeys.isPresent()) {
                return Optional.empty();
            }
            Set<VariableReferenceExpression> replicateVars = ImmutableSet.copyOf(unnest.getReplicateVariables());

            if (replicateVars.containsAll(resolvedKeys.get())
                    && isScanFilterProject(unnest.getSource())) {
                return Optional.of(unnest.getSource());
            }
        }

        // AggregationNode: clone the source if join keys are grouping keys
        if (peeled instanceof AggregationNode) {
            AggregationNode agg = (AggregationNode) peeled;
            Optional<Set<VariableReferenceExpression>> resolvedKeys = resolveVariablesThroughProjectFilter(node, peeled, joinKeyVars);
            if (!resolvedKeys.isPresent()) {
                return Optional.empty();
            }
            Set<VariableReferenceExpression> groupingKeys = ImmutableSet.copyOf(agg.getGroupingKeys());

            if (agg.getStep() == AggregationNode.Step.SINGLE
                    && agg.getGroupingSetCount() == 1
                    && !agg.hasEmptyGroupingSet()
                    && groupingKeys.containsAll(resolvedKeys.get())
                    && isScanFilterProject(agg.getSource())) {
                return Optional.of(agg.getSource());
            }
        }

        return Optional.empty();
    }

    /**
     * Traces join key variables backward through Filter/Project nodes
     * between the top node and the target node to find the source variables
     * they depend on. Returns empty if a projected expression is not a simple
     * variable reference (computed join keys cannot be safely traced).
     */
    private static Optional<Set<VariableReferenceExpression>> resolveVariablesThroughProjectFilter(
            PlanNode top,
            PlanNode target,
            Set<VariableReferenceExpression> vars)
    {
        if (top == target) {
            return Optional.of(vars);
        }

        // Build a chain of nodes from top to target
        PlanNode current = top;
        Set<VariableReferenceExpression> resolved = vars;

        while (current != target) {
            if (current instanceof ProjectNode) {
                ProjectNode project = (ProjectNode) current;
                ImmutableSet.Builder<VariableReferenceExpression> newResolved = ImmutableSet.builder();
                for (VariableReferenceExpression var : resolved) {
                    RowExpression expr = project.getAssignments().getMap().get(var);
                    if (expr instanceof VariableReferenceExpression) {
                        newResolved.add((VariableReferenceExpression) expr);
                    }
                    else if (expr != null) {
                        // Expression is not a simple variable reference;
                        // computed join keys cannot be safely traced
                        return Optional.empty();
                    }
                    else {
                        newResolved.add(var);
                    }
                }
                resolved = newResolved.build();
                current = project.getSource();
            }
            else if (current instanceof FilterNode) {
                current = ((FilterNode) current).getSource();
            }
            else {
                return Optional.empty();
            }
        }
        return Optional.of(resolved);
    }

    private static class Rewriter
            extends SimplePlanRewriter<Void>
    {
        private final Session session;
        private final Metadata metadata;
        private final PlanNodeIdAllocator idAllocator;
        private final VariableAllocator variableAllocator;
        private final FunctionAndTypeManager functionAndTypeManager;
        private final boolean complexEnabled;
        private final Optional<StatsProvider> stats;
        private boolean planChanged;

        private Rewriter(Session session, Metadata metadata, PlanNodeIdAllocator idAllocator, VariableAllocator variableAllocator, FunctionAndTypeManager functionAndTypeManager, Optional<StatsProvider> stats)
        {
            this.session = requireNonNull(session, "session is null");
            this.metadata = requireNonNull(metadata, "functionAndTypeManager is null");
            this.idAllocator = requireNonNull(idAllocator, "idAllocator is null");
            this.variableAllocator = requireNonNull(variableAllocator, "idAllocator is null");
            this.functionAndTypeManager = requireNonNull(functionAndTypeManager, "functionAndTypeManager is null");
            this.complexEnabled = isJoinPrefilterComplexBuildSideEnabled(session);
            this.stats = requireNonNull(stats, "stats is null");
        }

        @Override
        public PlanNode visitJoin(JoinNode node, RewriteContext<Void> context)
        {
            PlanNode left = node.getLeft();
            PlanNode right = node.getRight();

            PlanNode rewrittenLeft = rewriteWith(this, left);
            PlanNode rewrittenRight = rewriteWith(this, right);
            List<EquiJoinClause> equiJoinClause = node.getCriteria();

            if ((node.getType() == LEFT || node.getType() == INNER)
                    && !node.getCriteria().isEmpty()) {
                List<VariableReferenceExpression> leftKeyList = equiJoinClause.stream().map(EquiJoinClause::getLeft).collect(toImmutableList());
                List<VariableReferenceExpression> rightKeyList = equiJoinClause.stream().map(EquiJoinClause::getRight).collect(toImmutableList());

                // For an inner join, prefer filtering an aggregation's input to
                // cloning that input merely to filter the opposite scan. Keep
                // the original join: the prefilter must not change multiplicity.
                // A preserved outer-join side cannot be filtered this way.
                boolean filterLeft = complexEnabled && node.getType() == INNER
                        && canPrefilterAggregation(rewrittenLeft, leftKeyList)
                        && isScanFilterProjectOrUnion(rewrittenRight);
                PlanNode sourceSide = filterLeft ? rewrittenRight : rewrittenLeft;
                PlanNode targetSide = filterLeft ? rewrittenLeft : rewrittenRight;
                List<VariableReferenceExpression> sourceKeys = filterLeft ? rightKeyList : leftKeyList;
                List<VariableReferenceExpression> targetKeys = filterLeft ? leftKeyList : rightKeyList;
                Optional<PlanNode> cloneableSource = findCloneableSource(sourceSide, ImmutableSet.copyOf(sourceKeys), complexEnabled);
                if (cloneableSource.isPresent() && stats.isPresent()) {
                    // Estimate original inputs, which contain no variables
                    // introduced by this pass. Avoid cloning a full fact scan
                    // just to prefilter an already selective dimension result.
                    double sourceRows = stats.get().getStats(filterLeft ? right : left).getOutputRowCount();
                    double targetRows = stats.get().getStats(filterLeft ? left : right).getOutputRowCount();
                    // Unknown estimates (NaN) preserve the existing behavior.
                    if (sourceRows >= targetRows) {
                        cloneableSource = Optional.empty();
                    }
                }

                // The copy is refused when the subtree cannot be duplicated safely, e.g. it is not
                // deterministic, in which case the prefilter is not applied
                Map<VariableReferenceExpression, VariableReferenceExpression> leftVarMap = new HashMap<>();
                Optional<PlanNode> copiedLeftKeys = cloneableSource.flatMap(
                        source -> copyDeterministicScanNodes(source, metadata, idAllocator, sourceKeys, leftVarMap));

                if (copiedLeftKeys.isPresent()) {
                    checkState(IntStream.range(0, leftKeyList.size()).boxed().allMatch(i -> leftKeyList.get(i).getType().equals(rightKeyList.get(i).getType())));

                    boolean hashJoinKey = sourceKeys.size() > 1 || (sourceKeys.get(0).getType().equals(VARCHAR) || sourceKeys.get(0).getType() instanceof VarcharType);

                    // First create a SELECT DISTINCT leftKey FROM left
                    PlanNode leftKeys = copiedLeftKeys.get();
                    ImmutableList.Builder<RowExpression> expressionsToProject = ImmutableList.builder();
                    if (hashJoinKey) {
                        RowExpression hashExpression = getVariableHash(sourceKeys.stream().map(leftVarMap::get).collect(toImmutableList()), functionAndTypeManager);
                        expressionsToProject.add(hashExpression);
                    }
                    else {
                        expressionsToProject.add(leftVarMap.get(sourceKeys.get(0)));
                    }
                    PlanNode projectNode = projectExpressions(leftKeys, idAllocator, variableAllocator, expressionsToProject.build(), ImmutableList.of());

                    VariableReferenceExpression rightKeyToFilter = targetKeys.get(0);
                    RowExpression rightHashExpression = null;
                    if (hashJoinKey) {
                        rightHashExpression = getVariableHash(targetKeys, functionAndTypeManager);
                        rightKeyToFilter = variableAllocator.newVariable(rightHashExpression);
                    }

                    // DISTINCT on the leftkey or hash if wide column
                    PlanNode filteringSource = new AggregationNode(
                            sourceSide.getSourceLocation(),
                            idAllocator.getNextId(),
                            projectNode,
                            ImmutableMap.of(),
                            singleGroupingSet(projectNode.getOutputVariables()),
                            projectNode.getOutputVariables(),
                            AggregationNode.Step.SINGLE,
                            Optional.empty(),
                            Optional.empty(),
                            Optional.empty());

                    // There should be only one output variable. Project that
                    filteringSource = projectExpressions(filteringSource, idAllocator, variableAllocator, ImmutableList.of(filteringSource.getOutputVariables().get(0)), ImmutableList.of());

                    PlanNode filteredTarget = applyPrefilter(
                            targetSide,
                            filteringSource,
                            rightKeyToFilter,
                            targetKeys,
                            rightHashExpression,
                            hashJoinKey,
                            targetSide);
                    if (filterLeft) {
                        rewrittenLeft = filteredTarget;
                    }
                    else {
                        rewrittenRight = filteredTarget;
                    }
                }
            }

            if (rewrittenLeft != node.getLeft() || rewrittenRight != node.getRight()) {
                planChanged = true;
                return replaceChildren(node, ImmutableList.of(rewrittenLeft, rewrittenRight));
            }

            return node;
        }

        private static boolean canPrefilterAggregation(PlanNode node, List<VariableReferenceExpression> keys)
        {
            // Only identity projections preserve the key names used below the
            // aggregation. Computed or renamed keys require expression rewriting.
            while (node instanceof ProjectNode) {
                ProjectNode project = (ProjectNode) node;
                if (keys.stream().anyMatch(key -> !key.equals(project.getAssignments().get(key)))) {
                    return false;
                }
                node = project.getSource();
            }
            if (!(node instanceof AggregationNode)) {
                return false;
            }
            AggregationNode aggregation = (AggregationNode) node;
            return aggregation.getStep() == AggregationNode.Step.SINGLE
                    && aggregation.getGroupingSetCount() == 1
                    && !aggregation.hasEmptyGroupingSet()
                    && aggregation.getGroupingKeys().containsAll(keys);
        }

        private PlanNode applyPrefilter(
                PlanNode rewrittenRight,
                PlanNode filteringSource,
                VariableReferenceExpression rightKeyToFilter,
                List<VariableReferenceExpression> rightKeyList,
                RowExpression rightHashExpression,
                boolean hashJoinKey,
                PlanNode originalTarget)
        {
            // Try to push the prefilter below the target aggregation.
            if (complexEnabled) {
                Optional<PlanNode> pushed = tryPushPrefilterBelowAggregation(
                        rewrittenRight, filteringSource, rightKeyToFilter,
                        rightKeyList, rightHashExpression, hashJoinKey, originalTarget);
                if (pushed.isPresent()) {
                    return pushed.get();
                }
            }

            // Default: add prefilter on top of right side
            if (hashJoinKey) {
                rewrittenRight = addProjections(rewrittenRight, idAllocator, ImmutableMap.of(rightKeyToFilter, rightHashExpression));
            }

            VariableReferenceExpression semiJoinOutput = variableAllocator.newVariable("semiJoinOutput", BOOLEAN);
            SemiJoinNode semiJoinNode = new SemiJoinNode(
                    originalTarget.getSourceLocation(),
                    idAllocator.getNextId(),
                    Optional.empty(),
                    rewrittenRight,
                    filteringSource,
                    rightKeyToFilter,
                    filteringSource.getOutputVariables().get(0),
                    semiJoinOutput,
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty(),
                    ImmutableMap.of());

            PlanNode result = new FilterNode(semiJoinNode.getSourceLocation(), idAllocator.getNextId(), semiJoinNode, semiJoinOutput);
            if (result.getOutputVariables().size() > originalTarget.getOutputVariables().size()) {
                result = restrictOutput(result, idAllocator, originalTarget.getOutputVariables());
            }
            return result;
        }

        private Optional<PlanNode> tryPushPrefilterBelowAggregation(
                PlanNode rightSide,
                PlanNode filteringSource,
                VariableReferenceExpression rightKeyToFilter,
                List<VariableReferenceExpression> rightKeyList,
                RowExpression rightHashExpression,
                boolean hashJoinKey,
                PlanNode originalTarget)
        {
            if (!canPrefilterAggregation(rightSide, rightKeyList)) {
                return Optional.empty();
            }

            // Peel through identity-key Project nodes.
            PlanNode peeled = rightSide;
            ImmutableList.Builder<ProjectNode> projectStack = ImmutableList.builder();
            while (peeled instanceof ProjectNode) {
                projectStack.add((ProjectNode) peeled);
                peeled = ((ProjectNode) peeled).getSource();
            }

            if (!(peeled instanceof AggregationNode)) {
                return Optional.empty();
            }

            AggregationNode aggNode = (AggregationNode) peeled;
            Set<VariableReferenceExpression> groupingKeys = ImmutableSet.copyOf(aggNode.getGroupingKeys());

            if (aggNode.getStep() != AggregationNode.Step.SINGLE
                    || aggNode.getGroupingSetCount() != 1
                    || aggNode.hasEmptyGroupingSet()
                    || !groupingKeys.containsAll(rightKeyList)) {
                return Optional.empty();
            }

            // Build prefilter on the aggregation's source
            PlanNode aggSource = aggNode.getSource();
            if (hashJoinKey) {
                aggSource = addProjections(aggSource, idAllocator, ImmutableMap.of(rightKeyToFilter, rightHashExpression));
            }

            VariableReferenceExpression semiJoinOutput = variableAllocator.newVariable("semiJoinOutput", BOOLEAN);
            SemiJoinNode semiJoinNode = new SemiJoinNode(
                    originalTarget.getSourceLocation(),
                    idAllocator.getNextId(),
                    Optional.empty(),
                    aggSource,
                    filteringSource,
                    rightKeyToFilter,
                    filteringSource.getOutputVariables().get(0),
                    semiJoinOutput,
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty(),
                    ImmutableMap.of());

            PlanNode filtered = new FilterNode(semiJoinNode.getSourceLocation(), idAllocator.getNextId(), semiJoinNode, semiJoinOutput);

            // Restrict output to remove hash/semiJoin variables, keeping only the original agg source outputs
            filtered = restrictOutput(filtered, idAllocator, aggNode.getSource().getOutputVariables());

            // Rebuild the aggregation on top of the filtered source
            PlanNode result = new AggregationNode(
                    aggNode.getSourceLocation(),
                    idAllocator.getNextId(),
                    filtered,
                    aggNode.getAggregations(),
                    aggNode.getGroupingSets(),
                    aggNode.getPreGroupedVariables(),
                    aggNode.getStep(),
                    aggNode.getHashVariable(),
                    aggNode.getGroupIdVariable(),
                    aggNode.getAggregationId());

            // Rebuild any peeled Project nodes on top
            List<ProjectNode> projects = projectStack.build();
            for (int i = projects.size() - 1; i >= 0; i--) {
                ProjectNode proj = projects.get(i);
                result = new ProjectNode(
                        proj.getSourceLocation(),
                        idAllocator.getNextId(),
                        result,
                        proj.getAssignments(),
                        proj.getLocality());
            }

            return Optional.of(result);
        }

        public boolean isPlanChanged()
        {
            return planChanged;
        }
    }
}
