//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dylan Fauntleroy, Nathan Hales, Pranay Kumar, and some repeated code from John Miller
 *  @version 1.6
 *  @date    3/29/2021
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Project2 Code
 */

package scalation.analytics

import scala.collection.mutable.Set
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.noDouble
import scalation.plot.{Plot, PlotM}
import scalation.random.CDF.studentTCDF
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.util.banner
import scalation.util.Unicode.sub

import Fit._
import RegTechnique._

import Initializer._
import MatrixTransform._
import Optimizer._                                  // Optimizer - configuration
import Optimizer_SGD._                              // Stochastic Gradient Descent
//import Optimizer_SGDM._                               // Stochastic Gradient Descent with Momentum
import PredictorMat2._
import StoppingRule._
import ActivationFun._

//break


import scala.math.{max => MAX}

import scalation.linalgebra.{FunctionV_2V, MatriD, MatrixD, VectoD, VectorD, VectorI}
import scalation.plot.PlotM






//LassoTest10 tests Lasso regression on MLR for autompg
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `LassoRegressionTest10` object tests the `LassoRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 Bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.LassoRegressionTest10
 */
object LassoRegressionTest10 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg regression")
    val rg = new LassoRegression (x, y)
    println (rg.analyze ().report)
    println (rg.summary)
    val n = x.dim2                                                    // number of parameters/variables

    //banner ("Forward Selection Test")
    //val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv

    banner ("Forward Selection Test")
    val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for LassoRegression", lines = true)
    println (s"rSq = $rSq")

} // LassoRegressionTest10 object




//RidgeReg12 tests Ridge regression on MLR for autompg
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `RidgeRegressionTest12` object tests the `RidgeRegression` class using the AutoMPG
 *  dataset.  It illustrates using the `Relation` class for reading the data
 *  from a .csv file "auto-mpg.csv".  Assumes no missing values.
 *  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.RidgeRegressionTest12
 */
object RidgeRegressionTest12 extends App
{
    import scalation.columnar_db.Relation

    banner ("auto_mpg relation")
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()

    banner ("auto_mpg (x, y) dataset")
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")

    banner ("auto_mpg regression")
    val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
    println (rrg.analyze ().report)
    println (rrg.summary)
    val n = x.dim2                                                     // number of variables

    banner ("Forward Selection Test")
    val (cols, rSq) = rrg.forwardSelAll ()                             // R^2, R^2 bar, R^2 cv
    val k = cols.size
    val t = VectorD.range (1, k)                                       // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for RidgeRegression", lines = true)

    println (s"rSq = $rSq")

} // RidgeRegressionTest12 object


//TranRegressionTest8 tests forward selection on Tran regression for autompg
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegressionTest8` object tests the `TranRegression` class using the AutoMPG
 *  dataset.  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.TranRegressionTest8
 */
object TranRegressionTest8 extends App
{
    import ExampleAutoMPG._
    import TranRegression.{box_cox, cox_box}
    banner ("TranRegression feature selection - ExampleAutoMPG")

/*
    import scalation.columnar_db.Relation
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")
*/

//  val f = (log _ , exp _)                                        // try several transforms
//  val f = (sqrt _ , sq _)
//  val f = (sq _ , sqrt _)
    TranRegression.setLambda (0.2)                                 // try 0.2, 0.3, 0.4, 0.5, 0.6
    val f = (box_cox _ , cox_box _)

    TranRegression.rescaleOff ()
    banner (s"TranRegression with transform $f")
    val trg = TranRegression (ox, y, null, null, f._1, f._2, QR, null)    // automated
    println (trg.analyze ().report)
    println (trg.summary)

//  banner ("Cross-Validation Test")
//  trg.crossValidate ()

    banner ("Forward Selection Test")
    val (cols, rSq) = trg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv

    //banner ("Backward Elimination Test")
    //val (cols, rSq) = trg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv


    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for TranRegression", lines = true)

    println (s"rSq = $rSq")

} // TranRegressionTest8 object




//TranRegressionTest9 tests stepwise regression on Tran regression for autompg
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegressionTest9` object tests the `TranRegression` class using the AutoMPG
 *  dataset.  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.TranRegressionTest9
 */
object TranRegressionTest9 extends App
{
    import ExampleAutoMPG._
    import TranRegression.{box_cox, cox_box}
    banner ("TranRegression feature selection - ExampleAutoMPG")

/*
    import scalation.columnar_db.Relation
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")
*/

//  val f = (log _ , exp _)                                        // try several transforms
//  val f = (sqrt _ , sq _)
//  val f = (sq _ , sqrt _)
    TranRegression.setLambda (0.2)                                 // try 0.2, 0.3, 0.4, 0.5, 0.6
    val f = (box_cox _ , cox_box _)

    TranRegression.rescaleOff ()
    banner (s"TranRegression with transform $f")
    val trg = TranRegression (ox, y, null, null, f._1, f._2, QR, null)    // automated
    println (trg.analyze ().report)
    println (trg.summary)

//  banner ("Cross-Validation Test")
//  trg.crossValidate ()

    //banner ("Forward Selection Test")
    //val (cols, rSq) = trg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv

    //banner ("Backward Elimination Test")
    //val (cols, rSq) = trg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv

    banner ("Stepwise Regression Test")
    val (cols, rSq) = trg.stepRegressionAll ()   			// R^2, R^2 bar, R^2 cv


    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for TranRegression", lines = true)

    println (s"rSq = $rSq")

} // TranRegressionTest9 object




//TranRegressionTest10 tests forward selection on Tran regression for autompg
//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `TranRegressionTest10` object tests the `TranRegression` class using the AutoMPG
 *  dataset.  It also combines feature selection with cross-validation and plots
 *  R^2, R^2 bar and R^2 cv vs. the instance index.
 *  > runMain scalation.analytics.TranRegressionTest10
 */
object TranRegressionTest10 extends App
{
    import ExampleAutoMPG._
    import TranRegression.{box_cox, cox_box}
    banner ("TranRegression feature selection - ExampleAutoMPG")

/*
    import scalation.columnar_db.Relation
    val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
    auto_tab.show ()
    val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
    println (s"x = $x")
    println (s"y = $y")
*/

//  val f = (log _ , exp _)                                        // try several transforms
//  val f = (sqrt _ , sq _)
//  val f = (sq _ , sqrt _)
    TranRegression.setLambda (0.2)                                 // try 0.2, 0.3, 0.4, 0.5, 0.6
    val f = (box_cox _ , cox_box _)

    TranRegression.rescaleOff ()
    banner (s"TranRegression with transform $f")
    val trg = TranRegression (ox, y, null, null, f._1, f._2, QR, null)    // automated
    println (trg.analyze ().report)
    println (trg.summary)

//  banner ("Cross-Validation Test")
//  trg.crossValidate ()

    //banner ("Forward Selection Test")
    //val (cols, rSq) = trg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv

    banner ("Backward Elimination Test")
    val (cols, rSq) = trg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv

    //banner ("Stepwise Regression Test")
    //val (cols, rSq) = trg.stepRegressionAll ()   			// R^2, R^2 bar, R^2 cv


    val k = cols.size
    println (s"k = $k, n = ${x.dim2}")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for TranRegression", lines = true)

    println (s"rSq = $rSq")

} // TranRegressionTest10 object


//PerceptronTest10 tests forward selection on a perceptron trained model for autompg
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PerceptronTest10` object trains a perceptron on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.PerceptronTest10
 */
object PerceptronTest10 extends App
{
    import ExampleAutoMPG._
    banner ("Perceptron feature selection - ExampleAutoMPG")

    val f_ = f_sigmoid                                              // try different activation function
//  val f_ = f_tanh                                                 // try different activation function
//  val f_ = f_id                                                   // try different activation function
/*
    println ("ox = " + ox)
    println ("y  = " + y)
*/

    banner ("Perceptron with scaled y values")
    val hp2 = Optimizer.hp.updateReturn (("eta", 0.05), ("bSize", 10.0))
    val nn  = Perceptron (oxy, f0 = f_)                             // factory function automatically rescales
//  val nn  = new Perceptron (ox, y, f0 = f_)                       // constructor does not automatically rescale

    nn.train ().eval ()                                             // fit the weights using training data
    val n = ox.dim2                                                 // number of parameters/variables
    println (nn.report)
   
    banner ("Cross-Validation Test")
    nn.crossValidate ()

    banner ("Forward Selection Test")
    val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv


    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Perceptron", lines = true)

} // PerceptronTest10 object


//PerceptronTest11 tests backward elimination on a perceptron trained model for autompg
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PerceptronTest11` object trains a perceptron on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.PerceptronTest11
 */
object PerceptronTest11 extends App
{
    import ExampleAutoMPG._
    banner ("Perceptron feature selection - ExampleAutoMPG")

    val f_ = f_sigmoid                                              // try different activation function
//  val f_ = f_tanh                                                 // try different activation function
//  val f_ = f_id                                                   // try different activation function
/*
    println ("ox = " + ox)
    println ("y  = " + y)
*/

    banner ("Perceptron with scaled y values")
    val hp2 = Optimizer.hp.updateReturn (("eta", 0.05), ("bSize", 10.0))
    val nn  = Perceptron (oxy, f0 = f_)                             // factory function automatically rescales
//  val nn  = new Perceptron (ox, y, f0 = f_)                       // constructor does not automatically rescale

    nn.train ().eval ()                                             // fit the weights using training data
    val n = ox.dim2                                                 // number of parameters/variables
    println (nn.report)
   
    banner ("Cross-Validation Test")
    nn.crossValidate ()

    banner ("Backward Elimination Test")
    val (cols, rSq) = nn.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv


    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Perceptron", lines = true)

} // PerceptronTest11 object

//PerceptronTest12 tests stepwise regression on a perceptron trained model for autompg
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `PerceptronTest12` object trains a perceptron on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.PerceptronTest12
 */
object PerceptronTest12 extends App
{
    import ExampleAutoMPG._
    banner ("Perceptron feature selection - ExampleAutoMPG")

    val f_ = f_sigmoid                                              // try different activation function
//  val f_ = f_tanh                                                 // try different activation function
//  val f_ = f_id                                                   // try different activation function
/*
    println ("ox = " + ox)
    println ("y  = " + y)
*/

    banner ("Perceptron with scaled y values")
    val hp2 = Optimizer.hp.updateReturn (("eta", 0.05), ("bSize", 10.0))
    val nn  = Perceptron (oxy, f0 = f_)                             // factory function automatically rescales
//  val nn  = new Perceptron (ox, y, f0 = f_)                       // constructor does not automatically rescale

    nn.train ().eval ()                                             // fit the weights using training data
    val n = ox.dim2                                                 // number of parameters/variables
    println (nn.report)
   
    banner ("Cross-Validation Test")
    nn.crossValidate ()

    banner ("Stepwise Regression Test")
    val (cols, rSq) = nn.stepRegressionAll ()                          // R^2, R^2 bar, R^2 cv


    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for Perceptron", lines = true)

} // PerceptronTest12 object

//NeuralNet_3LTest7 trains a neural network with forward selection on AUTOMPG
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3LTest7` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.NeuralNet_3LTest7
 */
object NeuralNet_3LTest7 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_3L feature selection - ExampleAutoMPG")

    val f_ = (f_sigmoid, f_id)                                     // try different activation functions
//  val f_ = (f_tanh, f_id)                                        // try different activation functions
//  val f_ = (f_lreLU, f_id)                                       // try different activation functions

    banner ("NeuralNet_3L with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_3L (oxy, f0 = f_._1, f1 = f_._2)           // factory function automatically rescales
//  val nn  = new NeuralNet_3L (ox, y, f0 = f_._1, f1= f_._2)      // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Forward Selection Test")
    val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_3L", lines = true)

} // NeuralNet_3LTest7 object


//NeuralNet_3LTest8 trains a neural network with backward elimination on AUTOMPG
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3LTest8` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.NeuralNet_3LTest8
 */
object NeuralNet_3LTest8 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_3L feature selection - ExampleAutoMPG")

    val f_ = (f_sigmoid, f_id)                                     // try different activation functions
//  val f_ = (f_tanh, f_id)                                        // try different activation functions
//  val f_ = (f_lreLU, f_id)                                       // try different activation functions

    banner ("NeuralNet_3L with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_3L (oxy, f0 = f_._1, f1 = f_._2)           // factory function automatically rescales
//  val nn  = new NeuralNet_3L (ox, y, f0 = f_._1, f1= f_._2)      // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Backward Elimination Test")
    val (cols, rSq) = nn.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_3L", lines = true)

} // NeuralNet_3LTest8 object


//NeuralNet_3LTest9 trains a neural network with stepwise regression on AUTOMPG
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_3LTest9` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2.
 *  > runMain scalation.analytics.NeuralNet_3LTest9
 */
object NeuralNet_3LTest9 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_3L feature selection - ExampleAutoMPG")

    val f_ = (f_sigmoid, f_id)                                     // try different activation functions
//  val f_ = (f_tanh, f_id)                                        // try different activation functions
//  val f_ = (f_lreLU, f_id)                                       // try different activation functions

    banner ("NeuralNet_3L with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_3L (oxy, f0 = f_._1, f1 = f_._2)           // factory function automatically rescales
//  val nn  = new NeuralNet_3L (ox, y, f0 = f_._1, f1= f_._2)      // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Stepwise Regression Test")
    val (cols, rSq) = nn.stepRegressionAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_3L", lines = true)

} // NeuralNet_3LTest9 object


//NeuralNet_XLTest7 test forward selection on a neural net with multiple layers
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest7` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NeuralNet_XLTest7
 */
object NeuralNet_XLTest7 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
//  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
//  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Forward Selection Test")
    val (cols, rSq) = nn.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_XL", lines = true)

} // NeuralNet_XLTest7 object

//NeuralNet_XLTest8 tests stepwise regression on a neural net with multiple layers
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest8` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NeuralNet_XLTest8
 */
object NeuralNet_XLTest8 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
//  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
//  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Stepwise Regression Test")
    val (cols, rSq) = nn.stepRegressionAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_XL", lines = true)

} // NeuralNet_XLTest8 object


//NeuralNet_XLTest9 test backward elimination on a neural net with multiple layers
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `NeuralNet_XLTest9` object trains a neural network on the `ExampleAutoMPG` dataset.
 *  This tests forward feature/variable selection with plotting of R^2,
 *  > runMain scalation.analytics.NeuralNet_XLTest9
 */
object NeuralNet_XLTest9 extends App
{
    import ExampleAutoMPG._
    val n = ox.dim2                                                // number of parameters/variables
    banner ("NeuralNet_XL feature selection - ExampleAutoMPG")

    val af_ = Array (f_sigmoid, f_sigmoid, f_id)                   // try different activation functions
//  val af_ = Array (f_tanh, f_tanh, f_id)                         // try different activation functions

    banner ("NeuralNet_XL with scaled y values")
    hp("eta") = 0.02                                               // learning rate hyoer-parameter (see Optimizer)
    val nn  = NeuralNet_XL (oxy, af = af_)                         // factory function automatically rescales
//  val nn  = new NeuralNet_XL (ox, y, af = af_)                   // constructor does not automatically rescale

    nn.train ().eval ()                                            // fit the weights using training data
    println (nn.report)                                            // report parameters and fit
    val ft  = nn.fitA(0)                                           // fit for first output variable

    banner ("Backward Elimination Test")
    val (cols, rSq) = nn.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
    println (s"rSq = $rSq")
    val k = cols.size
    println (s"k = $k, n = $n")
    val t = VectorD.range (1, k)                                   // instance index
    new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
               "R^2 vs n for NeuralNet_XL", lines = true)

} // NeuralNet_XLTest9 object

