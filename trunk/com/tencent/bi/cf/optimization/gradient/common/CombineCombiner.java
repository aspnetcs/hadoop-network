package com.tencent.bi.cf.optimization.gradient.common;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;

import com.tencent.bi.utils.serialization.LongPairWritable;
import com.tencent.bi.utils.serialization.MatrixRowWritable;

public class CombineCombiner extends Reducer<LongPairWritable, MatrixRowWritable, LongPairWritable, MatrixRowWritable> {
	/**
	 * Output value
	 */
	private MatrixRowWritable outValue = new MatrixRowWritable();
			
	@Override
	public void reduce(LongPairWritable key, Iterable<MatrixRowWritable> values,
			Context context) throws IOException, InterruptedException {
		double[] res = null;
		for (MatrixRowWritable value: values) {	//combining
			double[] vec = value.viewVector();
			if(res==null) res = new double[vec.length];
			for (int i = 0; i < vec.length; i++)
				res[i] += vec[i];
		}
		outValue.set(res);
		context.write(key, outValue);
	}
}
