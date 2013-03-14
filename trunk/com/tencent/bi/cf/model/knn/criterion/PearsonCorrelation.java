package com.tencent.bi.cf.model.knn.criterion;

import java.util.List;
import java.util.Map;

/**
 * Pearson Correlation Criterion
 * @author purlin
 *
 */
public class PearsonCorrelation implements Criterion {

	@Override
	public double getValue(Map<Long, Double> vecA, Map<Long, Double> vecB, double lambda) {
		float meanA = 0.0f, meanB = 0.0f, cnt = 0.0f;
		float stdA = 0.0f, stdB = 0.0f, stdAB = 0.0f;
		for(Long k : vecA.keySet()){
			if(vecB.containsKey(k)){
				meanA += vecA.get(k);
				meanB += vecB.get(k);
				cnt ++;
			}
		}
		meanA /= cnt; meanB /= cnt;
		for(Long k : vecA.keySet()){
			if(vecB.containsKey(k)){
				double va = vecA.get(k);
				double vb = vecB.get(k);
				stdA += (va-meanA)*(va-meanA);
				stdB += (vb-meanB)*(vb-meanB);
				stdAB += (va-meanA)*(vb-meanB);
			}
		}
		if(stdA*stdB>0) return ((double) (cnt/(cnt+lambda))*(stdAB/(Math.sqrt(stdA*stdB))));
		else return 0.0f;
	}

	@Override
	public double getValue(List<Double> vecA, List<Double> vecB, double lambda) {
		double meanA = 0.0f, meanB = 0.0f, cnt = 0.0f;
		double stdA = 0.0f, stdB = 0.0f, stdAB = 0.0f;
		for(int i=0;i<vecA.size();i++){
			meanA += vecA.get(i);
			meanB += vecB.get(i);
			cnt ++;
		}
		meanA /= cnt; meanB /= cnt;
		for(int i=0;i<vecA.size();i++){
			double va = vecA.get(i);
			double vb = vecB.get(i);
			stdA += (va-meanA)*(va-meanA);
			stdB += (vb-meanB)*(vb-meanB);
			stdAB += (va-meanA)*(vb-meanB);
		}
		if(stdA*stdB>0) return (cnt/(cnt+lambda))*(stdAB/(Math.sqrt(stdA*stdB)));
		else return 0.0f;
	}
}
