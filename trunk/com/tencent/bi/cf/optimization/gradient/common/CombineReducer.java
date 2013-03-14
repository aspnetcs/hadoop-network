package com.tencent.bi.cf.optimization.gradient.common;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;

import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class CombineReducer extends Reducer<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
	/**
	 * Output value
	 */
	private MatrixRowWritable outValue = new MatrixRowWritable();
	
	@Override
	public void reduce(LongPairWritable key, Iterable<MatrixRowWritable> values,
			Context context) throws IOException, InterruptedException {
		double[] res = null;
		double pt = 0.0;
		for (MatrixRowWritable value: values) {	//combining
			double[] vec = value.viewVector();
			if(res==null) res = new double[vec.length-1];
			for (int i = 1; i < vec.length; i++)
				res[i-1] += vec[i];
			pt += vec[0];
		}
		for(int i=0;i<res.length;i++)
			res[i] /= pt;
		outValue.set(res);
		context.write(key, outValue);
	}
}
