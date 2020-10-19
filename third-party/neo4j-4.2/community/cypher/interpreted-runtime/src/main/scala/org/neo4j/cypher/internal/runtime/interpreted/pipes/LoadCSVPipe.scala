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
package org.neo4j.cypher.internal.runtime.interpreted.pipes

import java.net.URL

import org.neo4j.cypher.internal.ir.CSVFormat
import org.neo4j.cypher.internal.ir.HasHeaders
import org.neo4j.cypher.internal.ir.NoHeaders
import org.neo4j.cypher.internal.runtime.ArrayBackedMap
import org.neo4j.cypher.internal.runtime.ClosingIterator
import org.neo4j.cypher.internal.runtime.CypherRow
import org.neo4j.cypher.internal.runtime.QueryContext
import org.neo4j.cypher.internal.runtime.interpreted.commands.expressions.Expression
import org.neo4j.cypher.internal.util.attribution.Id
import org.neo4j.exceptions.LoadExternalResourceException
import org.neo4j.values.AnyValue
import org.neo4j.values.storable.TextValue
import org.neo4j.values.storable.Value
import org.neo4j.values.storable.Values
import org.neo4j.values.virtual.MapValueBuilder
import org.neo4j.values.virtual.VirtualValues

case class LoadCSVPipe(source: Pipe,
                       format: CSVFormat,
                       urlExpression: Expression,
                       variable: String,
                       fieldTerminator: Option[String],
                       legacyCsvQuoteEscaping: Boolean,
                       bufferSize: Int)
                      (val id: Id = Id.INVALID_ID)
  extends PipeWithSource(source) {

  protected def getImportURL(urlString: String, context: QueryContext): URL = {
    val url: URL = try {
      new URL(urlString)
    } catch {
      case e: java.net.MalformedURLException =>
        throw new LoadExternalResourceException(s"Invalid URL '$urlString': ${e.getMessage}", e)
    }

    context.getImportURL(url) match {
      case Left(error) =>
        throw new LoadExternalResourceException(s"Cannot load from URL '$urlString': $error")
      case Right(urlToLoad) =>
        urlToLoad
    }
  }

  private def copyWithLinenumber(filename: String, linenumber: Long, last: Boolean, row: CypherRow, key: String, value: AnyValue): CypherRow = {
    val newCtx = rowFactory.copyWith(row, key, value)
    newCtx.setLinenumber(filename, linenumber, last)
    newCtx
  }

  //Uses an ArrayBackedMap to store header-to-values mapping
  private class IteratorWithHeaders(headers: Seq[Value], context: CypherRow, filename: String, inner: LoadCsvIterator) extends ClosingIterator[CypherRow] {
    private val internalMap = new ArrayBackedMap[String, AnyValue](headers.map(a => if (a eq Values.NO_VALUE) null else a.asInstanceOf[TextValue].stringValue()).zipWithIndex.toMap)
    private var nextContext: CypherRow = _
    private var needsUpdate = true

    override protected[this] def closeMore(): Unit = inner.close()

    override def innerHasNext: Boolean = {
      if (needsUpdate) {
        nextContext = computeNextRow()
        needsUpdate = false
      }
      nextContext != null
    }

    override def next(): CypherRow = {
      if (!hasNext) Iterator.empty.next()
      needsUpdate = true
      nextContext
    }

    private def computeNextRow() = {
      if (inner.hasNext) {
        val row = inner.next().map(s => Values.ut8fOrNoValue(s))
        internalMap.putValues(row.asInstanceOf[Array[AnyValue]])
        //we need to make a copy here since someone may hold on this
        //reference, e.g. EagerPipe


        val builder = new MapValueBuilder
        for ((key, maybeNull) <- internalMap) {
          val value = if (maybeNull == null) Values.NO_VALUE else maybeNull
          builder.add(key, value)
        }
        copyWithLinenumber(filename, inner.lastProcessed, inner.readAll, context, variable, builder.build())
      } else null
    }
  }

  private class IteratorWithoutHeaders(context: CypherRow, filename: String, inner: LoadCsvIterator) extends ClosingIterator[CypherRow] {

    override protected[this] def closeMore(): Unit = inner.close()

    override def innerHasNext: Boolean = inner.hasNext

    override def next(): CypherRow = {
      // Make sure to pull on inner.next before calling inner.lastProcessed to get the right line number
      val value = VirtualValues.list(inner.next().map(s => Values.ut8fOrNoValue(s)): _*)
      copyWithLinenumber(filename, inner.lastProcessed, inner.readAll, context, variable, value)
    }
  }

  private def getLoadCSVIterator(state: QueryState, url: URL, useHeaders: Boolean): LoadCsvIterator ={
    state.resources.getCsvIterator(
      url, fieldTerminator, legacyCsvQuoteEscaping, bufferSize, useHeaders
    )
  }

  override protected def internalCreateResults(input: ClosingIterator[CypherRow], state: QueryState): ClosingIterator[CypherRow] = {
    input.flatMap(context => {
      val urlString: TextValue = urlExpression(context, state).asInstanceOf[TextValue]
      val url = getImportURL(urlString.stringValue(), state.query)

      format match {
        case HasHeaders =>
          val iterator = getLoadCSVIterator(state, url, useHeaders = true)
          val headers = if (iterator.nonEmpty) iterator.next().map(s => Values.ut8fOrNoValue(s)).toIndexedSeq else IndexedSeq.empty // First row is headers
          new IteratorWithHeaders(headers, context, url.getFile, iterator)
        case NoHeaders =>
          new IteratorWithoutHeaders(context, url.getFile, getLoadCSVIterator(state, url, useHeaders = false))
      }
    })
  }
}
