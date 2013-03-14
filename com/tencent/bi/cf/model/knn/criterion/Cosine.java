package com.tencent.bi.cf.model.knn.criterion;

import java.util.List;
import java.util.Map;


/**
 * Cosine criterion
 * @author purlin
 *
 */
public class Cosine implements Criterion {

	@Override
	public double getValue(Map<Long, Double> vecA, Map<Long, Double> vecB, double lambda) {
		float dot = 0.0f, modA = 0.0f, modB = 0.0f, cnt = 0.0f;
		for(Long k : vecA.keySet()){
			if(vecB.containsKey(k)){
				double va = vecA.get(k);
				double vb = vecB.get(k);
				dot += va*vb;
				modA += va*va; modB += vb*vb;
				cnt ++;
			}
		}
		return ((double) (cnt/(cnt+lambda))*(dot/Math.sqrt(modA*modB)));
	}
	
	@Override
	public double getValue(List<Double> vecA, List<Double> vecB, double lambda) {
		double dot = 0.0, modA = 0.0, modB = 0.0, cnt = 0.0;
		for(int i=0;i<vecA.size();i++){
			dot += vecA.get(i)*vecB.get(i);
			modA += vecA.get(i)*vecA.get(i); modB += vecB.get(i)*vecB.get(i);
			cnt ++;
		}
		return (cnt/(cnt+lambda))*(dot/Math.sqrt(modA*modB));
	}
	
}
