package com.tencent.bi.cf.optimization.gradient.common;

import java.io.IOException;
import java.util.Iterator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import com.tencent.bi.utils.StringUtils;

/**
 * Combiner reducer, for combining gradients.
 * @author tigerzhong
 *
 */
public class CombinerReducer extends Reducer<Text, Text, Text, Text> {
	/**
	 * Output value
	 */
	private Text outText = new Text();
	@Override
	public void reduce(Text key, Iterable<Text> values,
			Context context) throws IOException, InterruptedException {
		double[] res = null;
		//Number of single gradients.
		int pt = 0;
		Iterator<Text> it = values.iterator();
		while (it.hasNext()) {	//combining
			String line = it.next().toString();
			String[] tmp = line.split(",");
			if(pt==0) res = new double[tmp.length];
			for (int i = 0; i < tmp.length; i++)
			    // Sum up all gradients.
				res[i] += Double.parseDouble(tmp[i]);
			pt += 1;
		}
		for(int i=0;i<res.length;i++)
		    // Mean gradient
		    // @weixue Why mean?
		    // @weixue Code here is different from the document?
			res[i] /= pt;
		outText.set(StringUtils.array2String(res));
		context.write(key, outText);
	}
}
