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

import com.facebook.presto.Session;
import com.facebook.presto.spi.WarningCollector;
import com.facebook.presto.spi.plan.AggregationNode;
import com.facebook.presto.spi.plan.PlanNode;
import com.facebook.presto.sql.Optimizer;
import com.facebook.presto.sql.planner.assertions.BasePlanTest;
import com.google.common.collect.ImmutableList;
import org.testng.annotations.Test;

import static com.facebook.airlift.testing.Assertions.assertEqualsIgnoreOrder;
import static com.facebook.presto.SystemSessionProperties.REWRITE_CORRELATED_NOT_EQUAL_EXISTS;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

public class TestCorrelatedNotEqualExistsRewrite
        extends BasePlanTest
{
    private Session session(boolean enabled)
    {
        return Session.builder(getQueryRunner().getDefaultSession())
                .setSystemProperty(REWRITE_CORRELATED_NOT_EQUAL_EXISTS, Boolean.toString(enabled))
                .build();
    }

    private PlanNode plan(String sql, boolean enabled)
    {
        return getQueryRunner().inTransaction(session(enabled), transaction ->
                getQueryRunner().createPlan(transaction, sql, getQueryRunner().getPlanOptimizers(true),
                        Optimizer.PlanStage.OPTIMIZED_AND_VALIDATED, WarningCollector.NOOP).getRoot());
    }

    private static boolean hasSummary(PlanNode node)
    {
        if (node instanceof AggregationNode) {
            AggregationNode aggregation = (AggregationNode) node;
            if (aggregation.getAggregations().values().stream()
                    .anyMatch(value -> value.getCall().getDisplayName().equals("min"))) {
                return true;
            }
        }
        return node.getSources().stream().anyMatch(TestCorrelatedNotEqualExistsRewrite::hasSummary);
    }

    private void assertEquivalent(String sql)
    {
        assertEqualsIgnoreOrder(
                getQueryRunner().execute(session(true), sql).getMaterializedRows(),
                getQueryRunner().execute(session(false), sql).getMaterializedRows());
    }

    @Test
    public void testSummaryPlanAndDefaultDisabled()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.regionkey = n.regionkey AND n2.nationkey <> n.nationkey) FROM nation n";
        assertTrue(hasSummary(plan(sql, true)));
        assertFalse(hasSummary(plan(sql, false)));
        assertEquivalent(sql);
    }

    @Test
    public void testNullEmptyAndDuplicateInputs()
    {
        String outer = "(VALUES (1, BIGINT '1', BIGINT '10'), (1, BIGINT '1', BIGINT '10'), " +
                "(2, BIGINT '1', BIGINT '20'), (3, BIGINT '2', BIGINT '30'), " +
                "(4, BIGINT '3', BIGINT '40'), (5, BIGINT '4', BIGINT '10'), " +
                "(6, CAST(NULL AS BIGINT), BIGINT '10'), (7, BIGINT '1', CAST(NULL AS BIGINT))) o(id, k, v)";
        String inner = "(VALUES (BIGINT '1', BIGINT '10'), (BIGINT '1', BIGINT '20'), " +
                "(BIGINT '1', BIGINT '20'), (BIGINT '2', BIGINT '30'), " +
                "(BIGINT '3', CAST(NULL AS BIGINT)), (CAST(NULL AS BIGINT), BIGINT '20')) i(k, v)";
        String exists = "EXISTS (SELECT 1 FROM " + inner + " WHERE i.k = o.k AND i.v <> o.v)";
        String sql = "SELECT id, " + exists + " FROM " + outer;
        assertTrue(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
        assertEqualsIgnoreOrder(
                getQueryRunner().execute(session(true), sql).getMaterializedRows(),
                getQueryRunner().execute("VALUES (1, true), (1, true), (2, true), (3, false), " +
                        "(4, false), (5, false), (6, false), (7, false)").getMaterializedRows());
        assertEquivalent("SELECT id FROM " + outer + " WHERE " + exists);
        assertEquivalent("SELECT id FROM " + outer + " WHERE NOT " + exists);
        assertEquivalent("SELECT id, EXISTS (SELECT 1 FROM " + inner +
                " WHERE i.k = o.k AND i.v <> o.v AND i.k < 0) FROM " + outer);
    }

    @Test
    public void testCompositeKeysAndReversedComparison()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.regionkey = n.regionkey AND n2.name = n.name " +
                "AND n.nationkey <> n2.nationkey) FROM nation n";
        assertTrue(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
    }

    @Test
    public void testBigintExtremes()
    {
        String values = "(VALUES (BIGINT '1', BIGINT '-9223372036854775808'), " +
                "(BIGINT '1', BIGINT '9223372036854775807'), (BIGINT '2', BIGINT '0'), " +
                "(BIGINT '2', CAST(NULL AS BIGINT)), (BIGINT '3', BIGINT '9223372036854775807'))";
        String base = "SELECT o.v, EXISTS (SELECT 1 FROM " + values + " i(k, v) " +
                "WHERE i.k = o.k AND i.v <> o.v AND %s) FROM " + values + " o(k, v)";
        assertTrue(hasSummary(plan(String.format(base, "true"), true)));
        for (String predicate : ImmutableList.of("true", "i.v = BIGINT '0'", "i.v IS NULL", "false")) {
            assertEquivalent(String.format(base, predicate));
        }
    }

    @Test
    public void testNoEqualityKeyFallback()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.nationkey <> n.nationkey) FROM nation n";
        assertFalse(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
    }

    @Test
    public void testMultipleInequalitiesDoNotUseIndependentWitnesses()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.regionkey = n.regionkey AND n2.nationkey <> n.nationkey " +
                "AND n2.nationkey <> n.regionkey) FROM nation n";
        assertFalse(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
    }

    @Test
    public void testFloatingPointFallback()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.regionkey = n.regionkey " +
                "AND CAST(n2.nationkey AS DOUBLE) <> CAST(n.nationkey AS DOUBLE)) FROM nation n";
        assertFalse(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
    }

    @Test
    public void testLimitBeforeCorrelationFallback()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM " +
                "(SELECT nationkey, regionkey FROM nation ORDER BY nationkey LIMIT 3) n2 " +
                "WHERE n2.regionkey = n.regionkey AND n2.nationkey <> n.nationkey) FROM nation n";
        assertFalse(hasSummary(plan(sql, true)));
        assertEquivalent(sql);
    }

    @Test
    public void testNondeterministicInputFallback()
    {
        String sql = "SELECT n.nationkey, EXISTS (SELECT 1 FROM nation n2 " +
                "WHERE n2.regionkey = n.regionkey AND n2.nationkey <> n.nationkey " +
                "AND random() < 0.5) FROM nation n";
        assertFalse(hasSummary(plan(sql, true)));
    }
}
