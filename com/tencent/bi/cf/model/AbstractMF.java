package com.tencent.bi.cf.model;

import com.tencent.bi.cf.optimization.MFSolver;

import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.jet.math.Mult;

/**
 * Abstract Class for Matrix Factorization based Collaborative Filtering model builder..
 * @author tigerzhong
 *
 */
public abstract class AbstractMF implements CF{
	/**
	 * The matrix factorization solver used in optimization (the way we build
	 * the CF model).
	 */
	protected MFSolver solver = null;
	/**
	 * Number of iterations used in the optimization.
	 */
	protected int numIt = 100;
	/**
	 * Learning rate
	 */
	protected double learningRate = 0.01;
	/**
	 * Number of latent factors
	 */
	protected int numD = 5;
	/**
	 * Latent matrix for users. 1/2 of the MF-based CF model.
	 */
	protected DenseDoubleMatrix2D U = null;
	/**
	 * Latent matrix for items. 2/2 of the MF-based CF model.
	 */
	protected DenseDoubleMatrix2D V = null;
	/**
	 * Regularization parameter
	 */
	protected double lambda = 0.005;
	/**
	 * Input path of training Data 
	 */
	protected String inputPath = "";
	/**
	 * Class name for the MF {@link #solver} used.
	 */
	protected String solverName = "";
	
	/**
	 * Initialize this CF model builder.
	 * @param m number of users
	 * @param n number of items
	 * @param d number of latent dimensions
	 * @param solverName class name of the optimization method
	 * @param lambda reguralization parameter
	 * @param learningRate learning rate
	 * @throws Exception
	 */
	public void initModel(int m, int n, int d, String solverName, double lambda, double learningRate, int numIt, String inPath) throws Exception{
		//For the distributed version, we do not store the U and V in memory
		if(m>0)	{
			U = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(m,d);
			U.assign(Mult.div(d));
		}
		if(n>0) {
			V = (DenseDoubleMatrix2D) DoubleFactory2D.dense.random(n,d);
			V.assign(Mult.div(d));
		}
		//If the optimization method is the standard one for MF
		if(!solverName.equalsIgnoreCase(""))
			solver = (MFSolver) Class.forName(solverName).newInstance();
		this.lambda = lambda;
		this.learningRate = learningRate;
		this.numIt = numIt;
		this.inputPath = inPath;
		this.numD = d;
		this.solverName = solverName;
	}
		
	/**
	 * Predict a single rating
	 * @param p user id
	 * @param q item id
	 * @param o context id, such as position
	 * @return prediction value
	 */
	public abstract double predict(int p, int q, int o);
	
	/**
	 * Predict a single rating
	 * @param u latent vector for current user
	 * @param v latent vector for current item
	 * @param s latent vector for current context
	 * @return
	 */
	public abstract double predict(DoubleMatrix1D u, DoubleMatrix1D v, DoubleMatrix1D s);
	
	/**
	 * Set U
	 * @param u
	 */
	public void setU(DenseDoubleMatrix2D u) {
		U = u;
	}
	
	/**
	 * Set V
	 * @param v
	 */
	public void setV(DenseDoubleMatrix2D v) {
		V = v;
	}
}
