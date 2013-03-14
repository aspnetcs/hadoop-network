package com.tencent.bi.cf.model.knn.criterion;

import java.util.List;
import java.util.Map;

/**
 * Interface of criterion
 * @author purlin
 *
 */
public interface Criterion {
	/**
	 * Get similarity value of two objects
	 * @param vecA
	 * @param vecB
	 * @param lambda
	 * @return similarity value
	 */
	public double getValue(Map<Long, Double> vecA, Map<Long, Double> vecB, double lambda);
	/**
	 * Get similarity value of two objects
	 * @param vecA
	 * @param vecB
	 * @param lambda
	 * @return similarity value
	 */
	public double getValue(List<Double> vecA, List<Double> vecB, double lambda);
}
