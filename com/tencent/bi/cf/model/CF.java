package com.tencent.bi.cf.model;

/**
 * Interface for Collaborative Filtering model builder.
 * @author tigerzhong
 *
 */
public interface CF {
	/**
	 * Build Model
	 * @throws Exception
	 */
	public abstract void buildModel() throws Exception;
	/**
	 * Predict all user/item pairs with the data in inputPath.
	 * @param inputPath input path of data
	 * @param numD number of latent dimension
	 * @throws Exception
	 */
	public void predictPair(String inputPath, int numD) throws Exception;
	/**
	 * Predict all user/item pairs with the data supplied when {@link #buildModel()} was invoked.
	 * @throws Exception
	 */
	public void predictAll() throws Exception;
}
