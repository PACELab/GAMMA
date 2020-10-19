/*
 * Copyright (c) 2002-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
 * This file is part of Neo4j.
 *
 * Neo4j is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.neo4j.cypher.internal.compiler.planner.logical.steps

import org.neo4j.cypher.internal.ast.Hint
import org.neo4j.cypher.internal.ast.UsingIndexHint
import org.neo4j.cypher.internal.ast.semantics.SemanticTable
import org.neo4j.cypher.internal.compiler.IndexLookupUnfulfillableNotification
import org.neo4j.cypher.internal.compiler.planner.logical.LeafPlanFromExpressions
import org.neo4j.cypher.internal.compiler.planner.logical.LeafPlanner
import org.neo4j.cypher.internal.compiler.planner.logical.LeafPlansForVariable
import org.neo4j.cypher.internal.compiler.planner.logical.LeafPlansForVariable.maybeLeafPlans
import org.neo4j.cypher.internal.compiler.planner.logical.LogicalPlanningContext
import org.neo4j.cypher.internal.compiler.planner.logical.ordering.ResultOrdering
import org.neo4j.cypher.internal.compiler.planner.logical.ordering.ResultOrdering.PropertyAndPredicateType
import org.neo4j.cypher.internal.compiler.planner.logical.plans.AsDistanceSeekable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.AsPropertyScannable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.AsPropertySeekable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.AsStringRangeSeekable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.AsValueRangeSeekable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.PropertySeekable
import org.neo4j.cypher.internal.compiler.planner.logical.plans.Seekable
import org.neo4j.cypher.internal.expressions.Contains
import org.neo4j.cypher.internal.expressions.EndsWith
import org.neo4j.cypher.internal.expressions.Equals
import org.neo4j.cypher.internal.expressions.Expression
import org.neo4j.cypher.internal.expressions.FunctionInvocation
import org.neo4j.cypher.internal.expressions.FunctionName
import org.neo4j.cypher.internal.expressions.HasLabels
import org.neo4j.cypher.internal.expressions.LabelName
import org.neo4j.cypher.internal.expressions.LabelToken
import org.neo4j.cypher.internal.expressions.LogicalVariable
import org.neo4j.cypher.internal.expressions.PartialPredicate
import org.neo4j.cypher.internal.expressions.Property
import org.neo4j.cypher.internal.expressions.PropertyKeyName
import org.neo4j.cypher.internal.expressions.PropertyKeyToken
import org.neo4j.cypher.internal.expressions.Variable
import org.neo4j.cypher.internal.expressions.functions
import org.neo4j.cypher.internal.frontend.helpers.SeqCombiner
import org.neo4j.cypher.internal.ir.QueryGraph
import org.neo4j.cypher.internal.ir.ordering.InterestingOrder
import org.neo4j.cypher.internal.ir.ordering.ProvidedOrder
import org.neo4j.cypher.internal.logical.plans.AsDynamicPropertyNonSeekable
import org.neo4j.cypher.internal.logical.plans.AsStringRangeNonSeekable
import org.neo4j.cypher.internal.logical.plans.AsValueRangeNonSeekable
import org.neo4j.cypher.internal.logical.plans.CanGetValue
import org.neo4j.cypher.internal.logical.plans.CompositeQueryExpression
import org.neo4j.cypher.internal.logical.plans.ExistenceQueryExpression
import org.neo4j.cypher.internal.logical.plans.GetValueFromIndexBehavior
import org.neo4j.cypher.internal.logical.plans.IndexOrder
import org.neo4j.cypher.internal.logical.plans.IndexedProperty
import org.neo4j.cypher.internal.logical.plans.LogicalPlan
import org.neo4j.cypher.internal.logical.plans.QueryExpression
import org.neo4j.cypher.internal.logical.plans.SingleQueryExpression
import org.neo4j.cypher.internal.planner.spi.IndexDescriptor
import org.neo4j.cypher.internal.util.LabelId
import org.neo4j.cypher.internal.util.symbols.CTAny
import org.neo4j.cypher.internal.util.symbols.CypherType

abstract class AbstractIndexSeekLeafPlanner extends LeafPlanner with LeafPlanFromExpressions {

  // Abstract methods ***********

  protected def constructPlan(idName: String,
                              label: LabelToken,
                              properties: Seq[IndexedProperty],
                              isUnique: Boolean,
                              valueExpr: QueryExpression[Expression],
                              hint: Option[UsingIndexHint],
                              argumentIds: Set[String],
                              providedOrder: ProvidedOrder,
                              indexOrder: IndexOrder,
                              context: LogicalPlanningContext,
                              onlyExists: Boolean)
                             (solvedPredicates: Seq[Expression], predicatesForCardinalityEstimation: Seq[Expression]): LogicalPlan

  protected def findIndexesForLabel(labelId: Int, context: LogicalPlanningContext): Iterator[IndexDescriptor]

  // Concrete methods ***********

  override def producePlanFor(predicates: Set[Expression], qg: QueryGraph, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Set[LeafPlansForVariable] = {
    implicit val labelPredicateMap: Map[String, Set[HasLabels]] = qg.selections.labelPredicates
    implicit val semanticTable: SemanticTable = context.semanticTable
    if (labelPredicateMap.isEmpty)
      Set.empty
    else {
      val arguments: Set[LogicalVariable] = qg.argumentIds.map(n => Variable(n)(null))
      val indexCompatibles: Set[IndexCompatiblePredicate] = predicates.collect(
        asIndexCompatiblePredicate(qg.argumentIds, arguments, qg.hints))
      val result = indexCompatibles.map(_.name).flatMap { name =>
        val labelPredicates = labelPredicateMap.getOrElse(name, Set.empty)
        val nodePredicates = indexCompatibles.filter(p => p.name == name)
        maybeLeafPlans(name, producePlansForSpecificVariable(name, nodePredicates, labelPredicates, qg.hints, qg.argumentIds, context, interestingOrder))
      }

      if (result.isEmpty) {
        val seekableIdentifiers: Set[Variable] = findNonSeekableIdentifiers(qg.selections.flatPredicates, context)
        DynamicPropertyNotifier.process(seekableIdentifiers, IndexLookupUnfulfillableNotification, qg, context)
      }
      result
    }
  }

  override def apply(qg: QueryGraph, interestingOrder: InterestingOrder, context: LogicalPlanningContext): Seq[LogicalPlan] = {
    producePlanFor(qg.selections.flatPredicates.toSet, qg, interestingOrder, context).toSeq.flatMap(_.plans)
  }

  private def findNonSeekableIdentifiers(predicates: Seq[Expression], context: LogicalPlanningContext): Set[Variable] =
    predicates.flatMap {
      // n['some' + n.prop] IN [ ... ]
      case AsDynamicPropertyNonSeekable(nonSeekableId)
        if context.semanticTable.isNode(nonSeekableId) => Some(nonSeekableId)

      // n['some' + n.prop] STARTS WITH "prefix%..."
      case AsStringRangeNonSeekable(nonSeekableId)
        if context.semanticTable.isNode(nonSeekableId) => Some(nonSeekableId)

      // n['some' + n.prop] <|<=|>|>= value
      case AsValueRangeNonSeekable(nonSeekableId)
        if context.semanticTable.isNode(nonSeekableId) => Some(nonSeekableId)

      case _ => None
    }.toSet

  private def producePlansForSpecificVariable(idName: String,
                                              indexCompatiblePredicates: Set[IndexCompatiblePredicate],
                                              labelPredicates: Set[HasLabels],
                                              hints: Set[Hint], argumentIds: Set[String],
                                              context: LogicalPlanningContext,
                                              interestingOrder: InterestingOrder): Set[LogicalPlan] = {
    implicit val semanticTable: SemanticTable = context.semanticTable
    for (labelPredicate <- labelPredicates;
         labelName <- labelPredicate.labels;
         labelId: LabelId <- semanticTable.id(labelName).toSeq;
         indexDescriptor: IndexDescriptor <- findIndexesForLabel(labelId, context);
         predicatesForIndex <- predicatesForIndex(indexDescriptor, indexCompatiblePredicates, interestingOrder))
      yield
        createLogicalPlan(idName, hints, argumentIds, labelPredicate, labelName, labelId, predicatesForIndex, indexDescriptor.isUnique, context, semanticTable)
  }

  private def createLogicalPlan(idName: String,
                                hints: Set[Hint],
                                argumentIds: Set[String],
                                labelPredicate: HasLabels,
                                labelName: LabelName,
                                labelId: LabelId,
                                predicatesForIndex: PredicatesForIndex,
                                isUnique: Boolean,
                                context: LogicalPlanningContext,
                                semanticTable: SemanticTable): LogicalPlan = {
    val indexCompatiblePredicates = predicatesForIndex.predicatesInOrder.map(_.indexCompatiblePredicate)

    val hint = {
      val name = idName
      val propertyNames = indexCompatiblePredicates.map(_.propertyKeyName.name)
      hints.collectFirst {
        case hint@UsingIndexHint(Variable(`name`), `labelName`, propertyKeyName, _)
          if propertyKeyName.map(_.name) == propertyNames => hint
      }
    }

    val equalityPredicates = indexCompatiblePredicates.takeWhile(_.exactPredicate != NotExactPredicate)
    val equalityAndNextPredicates =
      if (equalityPredicates == indexCompatiblePredicates) equalityPredicates
      else equalityPredicates :+ indexCompatiblePredicates(equalityPredicates.length)
    val indexedPredicates: Seq[IndexCompatiblePredicate] =
      equalityAndNextPredicates ++ indexCompatiblePredicates.slice(equalityAndNextPredicates.length, indexCompatiblePredicates.length).map(_.convertToExists)

    val queryExpression: QueryExpression[Expression] = mergeQueryExpressionsToSingleOne(indexedPredicates)

    val properties = predicatesForIndex.predicatesInOrder.map { indexCompatiblePredicateWithGetValue =>
      val propertyName = indexCompatiblePredicateWithGetValue.indexCompatiblePredicate.propertyKeyName
      val getValue = indexCompatiblePredicateWithGetValue.getValueFromIndexBehavior
      IndexedProperty(PropertyKeyToken(propertyName, semanticTable.id(propertyName).head), getValue)
    }
    val entryConstructor: (Seq[Expression], Seq[Expression]) => LogicalPlan =
      constructPlan(idName, LabelToken(labelName, labelId), properties, isUnique, queryExpression, hint, argumentIds, predicatesForIndex.providedOrder, predicatesForIndex.indexOrder, context, indexedPredicates.head.isExists)

    val solvedPredicates = indexedPredicates.zip(indexCompatiblePredicates).filter(p => p._1 == p._2).map(_._1)
      .filter(_.solvesPredicate).map(p => p.propertyPredicate) :+ labelPredicate

    val predicatesForCardinalityEstimation = indexCompatiblePredicates.map(p => p.propertyPredicate) :+ labelPredicate
    entryConstructor(solvedPredicates, predicatesForCardinalityEstimation)
  }

  private def mergeQueryExpressionsToSingleOne(predicates: Seq[IndexCompatiblePredicate]): QueryExpression[Expression] =
    if (predicates.length == 1)
      predicates.head.queryExpression
    else {
      CompositeQueryExpression(predicates.map(_.queryExpression))
    }

  private def asIndexCompatiblePredicate(argumentIds: Set[String],
                                         arguments: Set[LogicalVariable],
                                         hints: Set[Hint])(
                                          implicit labelPredicateMap: Map[String, Set[HasLabels]],
                                          semanticTable: SemanticTable):
  PartialFunction[Expression, IndexCompatiblePredicate] = {
    def validDependencies(seekable: Seekable[_]): Boolean = {
      seekable.dependencies.forall(arguments) && !arguments(seekable.ident)
    }
    {
      // n.prop IN [ ... ]
      case predicate@AsPropertySeekable(seekable: PropertySeekable) if validDependencies(seekable) =>
        val queryExpression = seekable.args.asQueryExpression
        val exactPredicate = if (queryExpression.isInstanceOf[SingleQueryExpression[_]]) SingleExactPredicate else MultipleExactPredicate
        IndexCompatiblePredicate(seekable.name, seekable.propertyKey, predicate, queryExpression, seekable.propertyValueType(semanticTable), exactPredicate = exactPredicate,
          hints, argumentIds, solvesPredicate = true)

      // ... = n.prop
      // In some rare cases, we can't rewrite these predicates cleanly,
      // and so planning needs to search for these cases explicitly
      case predicate@Equals(a, prop@Property(seekable@LogicalVariable(_), propKeyName))
        if a.dependencies.forall(arguments) && !arguments(seekable) =>
        val expr = SingleQueryExpression(a)
        IndexCompatiblePredicate(seekable.name, propKeyName, predicate, expr, Seekable.cypherTypeForTypeSpec(semanticTable.getActualTypeFor(prop)), exactPredicate = SingleExactPredicate,
          hints, argumentIds, solvesPredicate = true)

      // n.prop STARTS WITH "prefix%..."
      case predicate@AsStringRangeSeekable(seekable) if validDependencies(seekable) =>
        val partialPredicate = PartialPredicate(seekable.expr, predicate)
        val queryExpression = seekable.asQueryExpression
        val propertyKey = seekable.propertyKey
        IndexCompatiblePredicate(seekable.name, propertyKey, partialPredicate, queryExpression, seekable.propertyValueType(semanticTable), exactPredicate = NotExactPredicate,
          hints, argumentIds, solvesPredicate = true)

      // n.prop <|<=|>|>= value
      case predicate@AsValueRangeSeekable(seekable) if validDependencies(seekable) =>
        val queryExpression = seekable.asQueryExpression
        val keyName = seekable.propertyKeyName
        IndexCompatiblePredicate(seekable.name, keyName, predicate, queryExpression, seekable.propertyValueType(semanticTable), exactPredicate = NotExactPredicate,
          hints, argumentIds, solvesPredicate = true)

      // The planned index seek will almost satisfy the predicate, but with the possibility of some false positives.
      // Since it reduces the cardinality to almost the level of the predicate, we can use the predicate to calculate cardinality,
      // but not mark it as solved, since the planner will still need to solve it with a Filter.
      case predicate@AsDistanceSeekable(seekable) if validDependencies(seekable) =>
        val queryExpression = seekable.asQueryExpression
        val keyName = seekable.propertyKeyName
        IndexCompatiblePredicate(seekable.name, keyName, predicate, queryExpression, seekable.propertyValueType(semanticTable), exactPredicate = NotExactPredicate,
          hints, argumentIds, solvesPredicate = false)

      // MATCH (n:User) WHERE exists(n.prop) RETURN n
      // Should only be allowed as part of an composite index:
      // "MATCH (n:User) WHERE n.foo = 'foo' AND exists(n.bar) RETURN n" with index on User(foo, bar)
      case predicate@AsPropertyScannable(scannable) if !arguments(scannable.ident) =>
        // scannable.expr is partialPredicate saying it solves exists() but not the predicate
        val solvedPredicate = if (scannable.solvesPredicate) predicate else scannable.expr
        IndexCompatiblePredicate(scannable.name, scannable.propertyKey, solvedPredicate, ExistenceQueryExpression(), CTAny,
          exactPredicate = NotExactPredicate, hints, argumentIds, solvesPredicate = true)

      // n.prop ENDS WITH 'substring'
      // It is always converted to exists and will need filtering
      case predicate@EndsWith(prop@Property(variable@Variable(name), keyName), expr) if expr.dependencies.forall(arguments) && !arguments(variable) =>
        // create a partialPredicate saying it solves exists() but not the ENDS WITH
        val solvedPredicate = PartialPredicate(
          FunctionInvocation(FunctionName(functions.Exists.name)(predicate.position), prop)(predicate.position),
          predicate
        )
        IndexCompatiblePredicate(name, keyName, solvedPredicate, ExistenceQueryExpression(), CTAny, exactPredicate = NotExactPredicate,
          hints, argumentIds, solvesPredicate = true)

      // n.prop CONTAINS 'substring'
      // It is always converted to exists and will need filtering
      case predicate@Contains(prop@Property(variable@Variable(name), keyName), expr) if expr.dependencies.forall(arguments) && !arguments(variable) =>
        // create a partialPredicate saying it solves exists() but not the CONTAINS
        val solvedPredicate = PartialPredicate(
          FunctionInvocation(FunctionName(functions.Exists.name)(predicate.position), prop)(predicate.position),
          predicate
        )
        IndexCompatiblePredicate(name, keyName, solvedPredicate, ExistenceQueryExpression(), CTAny, exactPredicate = NotExactPredicate,
          hints, argumentIds, solvesPredicate = true)
    }
  }

  private case class IndexCompatiblePredicateWithGetValue(indexCompatiblePredicate: IndexCompatiblePredicate, getValueFromIndexBehavior: GetValueFromIndexBehavior)
  private case class PredicatesForIndex(predicatesInOrder: Seq[IndexCompatiblePredicateWithGetValue], providedOrder: ProvidedOrder, indexOrder: IndexOrder)

  /**
   * Find and group all predicates, where one PredicatesForIndex contains one predicate for each indexed property, in the right order.
   */
  private def predicatesForIndex(indexDescriptor: IndexDescriptor, predicates: Set[IndexCompatiblePredicate], interestingOrder: InterestingOrder)
                                (implicit semanticTable: SemanticTable): Seq[PredicatesForIndex] = {

    // Group predicates by which property they include
    val predicatesByProperty = predicates
      .groupBy(icp => semanticTable.id(icp.propertyKeyName))
      // Sort out predicates that are not found in semantic table
      .collect { case (Some(x), v) => (x, v) }

    // For each indexed property, look up the relevant predicates
    val predicatesByIndexedProperty = indexDescriptor.properties
      .map(indexedProp => predicatesByProperty.getOrElse(indexedProp, Set.empty))

    // All combinations of predicates where each inner Seq covers the indexed properties in the correct order.
    // E.g. for an index on foo, bar and the predicates predFoo1, predFoo2, predBar1, this would return
    // Seq(Seq(predFoo1, predBar1), Seq(predFoo2, predBar1)).
    val matchingPredicateCombinations = SeqCombiner.combine(predicatesByIndexedProperty)

    matchingPredicateCombinations
      .map(matchingPredicates => matchPredicateWithIndexDescriptorAndInterestingOrder(matchingPredicates, indexDescriptor, interestingOrder))
  }

  private def matchPredicateWithIndexDescriptorAndInterestingOrder(matchingPredicates: Seq[IndexCompatiblePredicate],
                                                                   indexDescriptor: IndexDescriptor,
                                                                   interestingOrder: InterestingOrder): PredicatesForIndex = {
    val types = matchingPredicates.map(mp => mp.propertyType)

    // Ask the index for its value capabilities for the types of all properties.
    // We might override some of these later if they value is known in an equality predicate
    val propertyBehaviorFromIndex = indexDescriptor.valueCapability(types)

    // Combine plannable predicates with their available properties and getValueFromIndexBehavior
    val predicatesInOrder = propertyBehaviorFromIndex.zip(matchingPredicates).map {
      case (_, predicate) if predicate.exactPredicate != NotExactPredicate => IndexCompatiblePredicateWithGetValue(predicate, CanGetValue)
      case (behavior, predicate) => IndexCompatiblePredicateWithGetValue(predicate, behavior)
    }

    // Ask the index for its order capabilities for the types in prefix/subset defined by the interesting order
    val indexPropertiesAndPredicateTypes = matchingPredicates.map(mp => {
      val pos = mp.propertyPredicate.position
      PropertyAndPredicateType(Property(Variable(mp.name)(pos), mp.propertyKeyName)(pos), mp.exactPredicate == SingleExactPredicate)
    })

    val (providedOrder, indexOrder) = ResultOrdering.providedOrderForIndexOperator(interestingOrder, indexPropertiesAndPredicateTypes, types, indexDescriptor.orderCapability)

    PredicatesForIndex(predicatesInOrder, providedOrder, indexOrder)
  }

  /**
   * @param propertyType
   *                     We need to ask the index whether it supports getting values for that type
   * @param exactPredicate
   *                     We might already know if we can get values or not, for exact predicates. If this is `true` we will set GetValue,
   *                     otherwise we need to ask the index.
   */
  private case class IndexCompatiblePredicate(name: String,
                                              propertyKeyName: PropertyKeyName,
                                              propertyPredicate: Expression,
                                              queryExpression: QueryExpression[Expression],
                                              propertyType: CypherType,
                                              exactPredicate: ExactPredicate,
                                              hints: Set[Hint],
                                              argumentIds: Set[String],
                                              solvesPredicate: Boolean)
                                             (implicit labelPredicateMap: Map[String, Set[HasLabels]]) {

    def convertToExists: IndexCompatiblePredicate = queryExpression match {
      case _: ExistenceQueryExpression[Expression] => this
      case _: CompositeQueryExpression[Expression] => throw new IllegalStateException("A CompositeQueryExpression can't be nested in a CompositeQueryExpression")
      case _ =>
        copy(queryExpression = ExistenceQueryExpression(), exactPredicate = NotExactPredicate, solvesPredicate = false)
    }

    def isExists: Boolean = queryExpression match {
      case _: ExistenceQueryExpression[Expression] => true
      case _ => false
    }
  }

  sealed trait ExactPredicate
  case object SingleExactPredicate extends ExactPredicate
  case object MultipleExactPredicate extends ExactPredicate
  case object NotExactPredicate extends ExactPredicate
}
