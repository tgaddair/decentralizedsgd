package edu.stanford.taddair.DecentralizedSGD.examples

import akka.actor.{ActorSystem, Props}
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.Master.Start

object MainXOR extends App {
  println("XOR Training Example")

  val system = ActorSystem("XOR")
//  val xor = system.actorOf(Props(new CentralizedXOR))
  val xor = system.actorOf(Props(new DecentralizedXOR))

  xor ! Start
}
