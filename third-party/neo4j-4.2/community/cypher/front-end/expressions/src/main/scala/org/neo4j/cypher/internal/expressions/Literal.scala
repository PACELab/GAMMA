/*
 * Copyright (c) 2002-2020 "Neo4j,"
 * Neo4j Sweden AB [http://neo4j.com]
 *
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
package org.neo4j.cypher.internal.expressions

import java.util

import org.neo4j.cypher.internal.util.InputPosition

/**
 * Used for extracting the value from a Literal.
 *
 * Instead of using extensive instanceOf or match on every single literal
 * you can just do
 * {{{
 *   literal.writeTo(extractor)
 * }}}
 * and the method corresponding to the literal will be called on the provided extractor.
 */
trait LiteralExtractor {
  def writeBoolean(value: Boolean): Unit
  def writeNull(): Unit
  def writeString(value: String): Unit
  def writeDouble(value: Double): Unit
  def writeLong(value: Long): Unit
  def writeByteArray(value: Array[Byte]): Unit

  /**
   * Beginning of a list of size `size`
   *
   * It is guaranteed that the following `size` calls will be for the items
   * of the list in the order that they appear in the list.
   * @param size the size of the list
   */
  def beginList(size: Int): Unit

  /**
   * Called `size` calls after `beginList`
   */
  def endList(): Unit
}

trait LiteralWriter {
  def writeTo(extractor: LiteralExtractor)
}

sealed trait Literal extends Expression with LiteralWriter {
  def value: AnyRef
  def asCanonicalStringVal: String
}

sealed trait NumberLiteral extends Literal {
  def stringVal: String
  override def asCanonicalStringVal: String = stringVal
}

sealed trait IntegerLiteral extends NumberLiteral {
  def value: java.lang.Long
  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeLong(value)
}

sealed trait SignedIntegerLiteral extends IntegerLiteral
sealed trait UnsignedIntegerLiteral extends IntegerLiteral

sealed abstract class DecimalIntegerLiteral(stringVal: String) extends IntegerLiteral {
  lazy val value: java.lang.Long = java.lang.Long.parseLong(stringVal)
}

case class SignedDecimalIntegerLiteral(stringVal: String)(val position: InputPosition) extends DecimalIntegerLiteral(stringVal) with SignedIntegerLiteral
case class UnsignedDecimalIntegerLiteral(stringVal: String)(val position: InputPosition) extends DecimalIntegerLiteral(stringVal) with UnsignedIntegerLiteral

sealed abstract class OctalIntegerLiteral(stringVal: String) extends IntegerLiteral {
  lazy val value: java.lang.Long = java.lang.Long.decode(stringVal.toList.filter(c => c != 'o').mkString)
}

case class SignedOctalIntegerLiteral(stringVal: String)(val position: InputPosition) extends OctalIntegerLiteral(stringVal) with SignedIntegerLiteral

sealed abstract class HexIntegerLiteral(stringVal: String) extends IntegerLiteral {
  lazy val value: java.lang.Long = java.lang.Long.decode(stringVal)
}

case class SignedHexIntegerLiteral(stringVal: String)(val position: InputPosition) extends HexIntegerLiteral(stringVal) with SignedIntegerLiteral


sealed trait DoubleLiteral extends NumberLiteral {
  def value: java.lang.Double
  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeDouble(value)
}

case class DecimalDoubleLiteral(stringVal: String)(val position: InputPosition) extends DoubleLiteral {
  lazy val value: java.lang.Double = java.lang.Double.parseDouble(stringVal)
}

case class StringLiteral(value: String)(val position: InputPosition) extends Literal {
  override def asCanonicalStringVal = value

  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeString(value)
}

final case class SensitiveStringLiteral(value: Array[Byte])(val position: InputPosition) extends Expression with SensitiveString with LiteralWriter {
  override def equals(obj: Any): Boolean = obj match {
    case o: SensitiveStringLiteral => util.Arrays.equals(o.value, value)
    case _ => false
  }

  override def hashCode(): Int = util.Arrays.hashCode(value)

  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeByteArray(value)
}

trait SensitiveString {
  val position: InputPosition
}

case class Null()(val position: InputPosition) extends Literal {
  val value = null

  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeNull()
  override def asCanonicalStringVal = "NULL"
}

object Null {
  val NULL: Null = Null()(InputPosition.NONE)
}

sealed trait BooleanLiteral extends Literal

case class True()(val position: InputPosition) extends BooleanLiteral {
  val value: java.lang.Boolean = true

  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeBoolean(true)
  override def asCanonicalStringVal = "true"
}

case class False()(val position: InputPosition) extends BooleanLiteral {
  val value: java.lang.Boolean = false
  override def writeTo(extractor: LiteralExtractor): Unit = extractor.writeBoolean(false)
  override def asCanonicalStringVal = "false"
}
