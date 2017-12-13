package edu.stanford.taddair.DecentralizedSGD.actors.centralized

import akka.actor.{Actor, ActorLogging}
import breeze.linalg.DenseVector
import edu.stanford.taddair.DecentralizedSGD.actors.centralized.OutputActor.{Output, ParametersUpdated}

object OutputActor {

  case class Output(replicaId: Int, target: DenseVector[Double], output: DenseVector[Double])

  case class ParametersUpdated(count: Int)

}

/**
 * Actor that logs outputs and keeps track of the last predictions of each model replica
 */
class OutputActor extends Actor with ActorLogging {

  var updates = 0;

  def receive = {

    case Output(replica, target, output) => {

//      log.info(s"replica id ${replica}, output: ${output}, target ${target}")
    }

    case ParametersUpdated(count) => {
      updates += count;
      log.info(s"parameters updated: ${updates}")
    }


  }


}
