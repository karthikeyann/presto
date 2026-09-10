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
import com.facebook.presto.spi.plan.AggregationNode;
import com.facebook.presto.spi.plan.JoinNode;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.spi.plan.SemiJoinNode;
import com.facebook.presto.sql.planner.assertions.BasePlanTest;
import com.google.common.collect.ImmutableList;
import org.testng.annotations.Test;

import static com.facebook.airlift.testing.Assertions.assertEqualsIgnoreOrder;
import static com.facebook.presto.SystemSessionProperties.JOIN_PREFILTER_BUILD_SIDE;
import static com.facebook.presto.SystemSessionProperties.JOIN_PREFILTER_COMPLEX_BUILD_SIDE;
import static com.facebook.presto.SystemSessionProperties.JOIN_PREFILTER_COST_BASED;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

public class TestJoinPrefilter
        extends BasePlanTest
{
    private Session enableBasic()
    {
        return Session.builder(getQueryRunner().getDefaultSession())
                .setSystemProperty(JOIN_PREFILTER_BUILD_SIDE, "true")
                .build();
    }

    private Session enableComplex()
    {
        return Session.builder(getQueryRunner().getDefaultSession())
                .setSystemProperty(JOIN_PREFILTER_BUILD_SIDE, "true")
                .setSystemProperty(JOIN_PREFILTER_COMPLEX_BUILD_SIDE, "true")
                .build();
    }

    private Session enableComplexWithoutJoinReordering()
    {
        return Session.builder(enableComplex())
                .setSystemProperty("join_reordering_strategy", "NONE")
                .build();
    }

    private void assertSameResultsWithPrefilter(String sql)
    {
        assertSameResultsWithPrefilter(sql, enableComplexWithoutJoinReordering());
    }

    private void assertSameResultsWithPrefilter(String sql, Session enabled)
    {
        Session disabled = Session.builder(enabled)
                .setSystemProperty(JOIN_PREFILTER_BUILD_SIDE, "false")
                .build();
        assertEqualsIgnoreOrder(
                getQueryRunner().execute(enabled, sql).getMaterializedRows(),
                getQueryRunner().execute(disabled, sql).getMaterializedRows());
    }

    private boolean hasSemiJoinBelowLeftSideAggregation(String sql, Session session)
    {
        JoinNode join = findFirst(getOptimizedPlan(sql, session), JoinNode.class);
        return join != null && hasAggregationWithSemiJoinBelow(join.getLeft());
    }

    private PlanNode getOptimizedPlan(String sql, Session session)
    {
        return getQueryRunner().inTransaction(session, transactionSession ->
                getQueryRunner().createPlan(
                        transactionSession,
                        sql,
                        getQueryRunner().getPlanOptimizers(true),
                        com.facebook.presto.sql.Optimizer.PlanStage.OPTIMIZED_AND_VALIDATED,
                        com.facebook.presto.spi.WarningCollector.NOOP).getRoot());
    }

    private boolean planContainsSemiJoin(String sql, Session session)
    {
        return containsNode(getOptimizedPlan(sql, session), SemiJoinNode.class);
    }

    /**
     * Checks if any SemiJoinNode appears as a descendant of an AggregationNode
     * on the right side of the top-level join. This verifies the prefilter
     * was pushed below the right-side aggregation.
     */
    private boolean hasSemiJoinBelowRightSideAggregation(String sql, Session session)
    {
        PlanNode root = getOptimizedPlan(sql, session);
        // Find the top-level join
        JoinNode join = findFirst(root, JoinNode.class);
        if (join == null) {
            return false;
        }
        // Check if right side has aggregation with SemiJoin below it
        return hasAggregationWithSemiJoinBelow(join.getRight());
    }

    private static boolean hasAggregationWithSemiJoinBelow(PlanNode node)
    {
        if (node instanceof AggregationNode) {
            return containsNode(node, SemiJoinNode.class);
        }
        return node.getSources().stream().anyMatch(
                source -> hasAggregationWithSemiJoinBelow(source));
    }

    @SuppressWarnings("unchecked")
    private static <T extends PlanNode> T findFirst(PlanNode node, Class<T> nodeClass)
    {
        if (nodeClass.isInstance(node)) {
            return (T) node;
        }
        for (PlanNode source : node.getSources()) {
            T found = findFirst(source, nodeClass);
            if (found != null) {
                return found;
            }
        }
        return null;
    }

    private static boolean containsNode(PlanNode node, Class<?> nodeClass)
    {
        if (nodeClass.isInstance(node)) {
            return true;
        }
        return node.getSources().stream().anyMatch(source -> containsNode(source, nodeClass));
    }

    @Test
    public void testBasicScanFilterProject()
    {
        assertTrue(planContainsSemiJoin(
                "SELECT * FROM nation n JOIN region r ON n.regionkey = r.regionkey",
                enableBasic()));
    }

    @Test
    public void testCostBasedSkipsLargerFilteringSource()
    {
        Session enabled = Session.builder(enableComplexWithoutJoinReordering())
                .setSystemProperty(JOIN_PREFILTER_COST_BASED, "true")
                .build();
        String sql = "SELECT * FROM nation n JOIN region r ON n.regionkey = r.regionkey";
        assertFalse(planContainsSemiJoin(sql, enabled));
        assertTrue(planContainsSemiJoin(sql, enableComplexWithoutJoinReordering()));
        assertSameResultsWithPrefilter(sql, enabled);
    }

    @Test
    public void testCostBasedKeepsSmallerFilteringSource()
    {
        Session enabled = Session.builder(enableComplexWithoutJoinReordering())
                .setSystemProperty(JOIN_PREFILTER_COST_BASED, "true")
                .build();
        String sql = "SELECT * FROM region r JOIN nation n ON r.regionkey = n.regionkey";
        assertTrue(planContainsSemiJoin(sql, enabled));
        assertSameResultsWithPrefilter(sql, enabled);
    }

    @Test
    public void testUnionAllLeft()
    {
        assertTrue(planContainsSemiJoin(
                "SELECT * FROM " +
                        "(SELECT regionkey FROM nation UNION ALL SELECT regionkey FROM nation) t " +
                        "JOIN region r ON t.regionkey = r.regionkey",
                enableComplex()));
    }

    @Test
    public void testCrossJoinLeft()
    {
        // Use join_reordering_strategy=NONE to prevent the cross join from being reordered
        Session session = Session.builder(getQueryRunner().getDefaultSession())
                .setSystemProperty(JOIN_PREFILTER_BUILD_SIDE, "true")
                .setSystemProperty(JOIN_PREFILTER_COMPLEX_BUILD_SIDE, "true")
                .setSystemProperty("join_reordering_strategy", "NONE")
                .build();
        assertTrue(planContainsSemiJoin(
                "SELECT * FROM " +
                        "(SELECT n.regionkey, r.name AS rname FROM nation n CROSS JOIN region r) t " +
                        "JOIN region r2 ON t.regionkey = r2.regionkey",
                session));
    }

    @Test
    public void testAggregationLeft()
    {
        assertTrue(planContainsSemiJoin(
                "SELECT * FROM " +
                        "(SELECT regionkey, count(*) AS cnt FROM nation GROUP BY regionkey) t " +
                        "JOIN region r ON t.regionkey = r.regionkey",
                enableComplex()));
    }

    @Test
    public void testAggregationLeftPushdown()
    {
        String sql = "SELECT t.regionkey, t.cnt, n.nationkey FROM " +
                "(SELECT regionkey, count(*) AS cnt FROM nation GROUP BY regionkey) t " +
                "JOIN nation n ON t.regionkey = n.regionkey WHERE n.nationkey < 10";
        assertTrue(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
        assertFalse(hasSemiJoinBelowLeftSideAggregation(sql, enableBasic()));
        // Multiple matching build rows must still duplicate the aggregate row.
        assertSameResultsWithPrefilter(sql);
    }

    @Test
    public void testAggregationLeftPushdownMultipleKeys()
    {
        String sql = "SELECT t.regionkey, t.name, t.cnt FROM " +
                "(SELECT regionkey, name, count(*) AS cnt FROM nation GROUP BY regionkey, name) t " +
                "JOIN nation n ON t.regionkey = n.regionkey AND t.name = n.name WHERE n.nationkey < 10";
        assertTrue(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
        assertSameResultsWithPrefilter(sql);
    }

    @Test
    public void testAggregationPrefilterAfterJoinReordering()
    {
        String sql = "SELECT sum(l.extendedprice) FROM lineitem l " +
                "JOIN part p ON l.partkey = p.partkey " +
                "JOIN (SELECT partkey, avg(quantity) AS average_quantity FROM lineitem GROUP BY partkey) a " +
                "ON a.partkey = p.partkey " +
                "WHERE p.brand = 'Brand#23' AND p.container = 'MED BOX' " +
                "AND l.quantity < 0.2 * a.average_quantity";
        PlanNode plan = getOptimizedPlan(sql, enableComplex());
        assertTrue(hasGroupedAggregationWithSemiJoinBelow(plan));
        assertSameResultsWithPrefilter(sql, enableComplex());
    }

    private static boolean hasGroupedAggregationWithSemiJoinBelow(PlanNode node)
    {
        if (node instanceof AggregationNode && !((AggregationNode) node).getGroupingKeys().isEmpty()) {
            return containsNode(node, SemiJoinNode.class);
        }
        return node.getSources().stream().anyMatch(TestJoinPrefilter::hasGroupedAggregationWithSemiJoinBelow);
    }

    @Test
    public void testAggregationLeftPushdownNullAndEmptyKeys()
    {
        String aggregate = "(SELECT k, count(*) AS cnt FROM " +
                "(SELECT IF(nationkey = 0, CAST(NULL AS BIGINT), regionkey) k FROM nation) " +
                "GROUP BY k) t";
        for (String predicate : ImmutableList.of("nationkey < 10", "nationkey < 0")) {
            String sql = "SELECT t.k, t.cnt, n.nationkey FROM " + aggregate +
                    " JOIN (SELECT IF(nationkey = 1, CAST(NULL AS BIGINT), regionkey) k, nationkey " +
                    "FROM nation WHERE " + predicate + ") n ON t.k = n.k";
            assertSameResultsWithPrefilter(sql);
        }
    }

    @Test
    public void testAggregationLeftNoPushdownForOuterJoin()
    {
        String sql = "SELECT t.regionkey, t.cnt, n.nationkey FROM " +
                "(SELECT regionkey, count(*) AS cnt FROM nation GROUP BY regionkey) t " +
                "LEFT JOIN (SELECT regionkey, nationkey FROM nation WHERE nationkey < 2) n " +
                "ON t.regionkey = n.regionkey";
        assertFalse(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
        assertSameResultsWithPrefilter(sql);
    }

    @Test
    public void testAggregationLeftNoPushdownForNonGroupingKey()
    {
        String sql = "SELECT t.nationkey, t.regionkey FROM " +
                "(SELECT nationkey, max(regionkey) AS regionkey FROM nation GROUP BY nationkey) t " +
                "JOIN nation n ON t.regionkey = n.regionkey WHERE n.nationkey < 10";
        assertFalse(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
        assertSameResultsWithPrefilter(sql);
    }

    @Test
    public void testAggregationLeftNoPushdownForGroupingSets()
    {
        String sql = "SELECT t.regionkey, t.cnt FROM " +
                "(SELECT regionkey, count(*) AS cnt FROM nation " +
                "GROUP BY GROUPING SETS ((regionkey), (regionkey, nationkey))) t " +
                "JOIN nation n ON t.regionkey = n.regionkey WHERE n.nationkey < 10";
        assertFalse(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
        assertSameResultsWithPrefilter(sql);
    }

    @Test
    public void testAggregationLeftNoPushdownForNonDeterministicBuild()
    {
        String sql = "SELECT t.regionkey, t.cnt FROM " +
                "(SELECT regionkey, count(*) AS cnt FROM nation GROUP BY regionkey) t " +
                "JOIN (SELECT regionkey FROM nation WHERE random() > 0.5) n ON t.regionkey = n.regionkey";
        assertFalse(hasSemiJoinBelowLeftSideAggregation(sql, enableComplexWithoutJoinReordering()));
    }

    @Test
    public void testAggregationRightPushdown()
    {
        // Verify the SemiJoin is pushed below the right-side aggregation
        assertTrue(hasSemiJoinBelowRightSideAggregation(
                "SELECT * FROM nation n " +
                        "JOIN (SELECT regionkey, count(*) AS cnt FROM nation GROUP BY regionkey) t " +
                        "ON n.regionkey = t.regionkey",
                enableComplex()));
    }

    @Test
    public void testUnnestLeft()
    {
        assertTrue(planContainsSemiJoin(
                "SELECT * FROM " +
                        "(SELECT n.regionkey, x FROM nation n CROSS JOIN UNNEST(ARRAY[1,2,3]) t(x)) t " +
                        "JOIN region r ON t.regionkey = r.regionkey",
                enableComplex()));
    }

    @Test
    public void testComplexDisabledWithoutFlag()
    {
        // Keep the union on the probe side: reordering may put a basic scan
        // there instead, in which case the basic prefilter is applicable.
        Session session = Session.builder(enableBasic())
                .setSystemProperty("join_reordering_strategy", "NONE")
                .build();
        assertFalse(planContainsSemiJoin(
                "SELECT * FROM " +
                        "(SELECT regionkey FROM nation UNION ALL SELECT regionkey FROM nation) t " +
                        "JOIN region r ON t.regionkey = r.regionkey",
                session));
    }

    @Test
    public void testAggregationRightNoPushdownForGroupingSets()
    {
        // Multiple grouping sets: SemiJoin should NOT be pushed below the aggregation
        assertFalse(hasSemiJoinBelowRightSideAggregation(
                "SELECT * FROM nation n " +
                        "JOIN (" +
                        "   SELECT regionkey, count(*) AS cnt " +
                        "   FROM nation " +
                        "   GROUP BY GROUPING SETS ((regionkey), (regionkey, nationkey))" +
                        ") t " +
                        "ON n.regionkey = t.regionkey",
                enableComplex()));
    }

    @Test
    public void testAggregationRightNoPushdownWhenJoinKeyNotGroupingKey()
    {
        // Right-side join key (regionkey) is a max() output, not a grouping key
        assertFalse(hasSemiJoinBelowRightSideAggregation(
                "SELECT * FROM nation n " +
                        "JOIN (" +
                        "   SELECT nationkey, max(regionkey) AS regionkey, count(*) AS cnt " +
                        "   FROM nation " +
                        "   GROUP BY nationkey" +
                        ") t " +
                        "ON n.regionkey = t.regionkey",
                enableComplex()));
    }
}
