//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Dylan Fauntleroy, with repeated code from Hao Peng and John Miller
 *  @version 1.6
 *  @date    Tue Sep 29 14:14:15 EDT 2015
 *  @see     LICENSE (MIT style license file).
 *
 *  @title   Example Dataset: Red bikes quality
 */

package scalation.analytics

import scalation.linalgebra.{MatrixD, VectorD, vectorD2StatVec}
import scalation.plot.Plot
import scalation.util.banner


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The `StabilityStuff` object stored the UCI Stability dataset in a matrix.
 *  @see archive.ics.uci.edu/ml/datasets/Auto+MPG
 */
object StabilityStuff
{
    /** the names of the predictor variables and the response variable at the end
     */
    //val fnamer = Array ("cylinders", "displacement", "horsepower", "weight", "acceleration",
                        //"model_year", "origin", "mpg")

	val fnamer = Array("tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4", "stab")


    val fname = Array("tau1", "tau2", "tau3", "tau4", "p1", "p2", "p3", "p4", "g1", "g2", "g3", "g4", "stab")

    /** the raw combined data matrix 'xyr'
     */

   import scalation.columnar_db.Relation
banner ("Stability relation")
    val auto_tab = Relation (BASE_DIR + "Stability_Data.csv", "Stability", null, -1)
    //val auto_tab = Relation.apply(BASE_DIR + "Stability_Data.csv", "Stability", 0,  null,  ",", null)
    //auto_tab.show ()

    banner ("Stability (x, y) dataset")
    val xyr = auto_tab.toMatriD (0 to 12)
    //println (s"x = $x")
    //println (s"y = $y")

    //val xyr = new MatrixD ((1031,  ), 540,0,0,162,2.5,1040,676,28,79.99,





    //val xy = xyr.sliceEx (xyr.dim1, 6)                                 // remove the origin column
    val xy = xyr                                                       // use all columns - may cause multi-collinearity

    /** vector of all ones
     */
    val _1 = VectorD.one (xy.dim1)

    /** index for the data points (instances)
     */
    val t = VectorD.range (0, xy.dim1)

    /** the separation of the combine data matrix 'xy' into
     *  a data/input matrix 'x' and a response/output vector 'y'
     */
    val (x, y) = pullResponse (xy)

    /** the combined data matrix 'xy' with a column of all ones prepended
     *  for intercept models
     */
    val oxy = _1 +^: xy

    /** the data matrix 'x' with a column of all ones prepended for intercept models
     */
    val ox = _1 +^: x

} // StabilityStuff object


